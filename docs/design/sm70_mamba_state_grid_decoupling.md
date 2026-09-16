# SM70 Mamba state grid: decoupling from the KV block size

Date: 2026-09-17

Target: Qwen3.8-27B NVFP4 (QUASAR QAT) on 4x V100-SXM2-32GB, TP4, fp8_e4m3 KV,
DFlash2 with 7 drafts, prefix caching on, `--mamba-cache-mode align`,
`--max-num-batched-tokens 8192 --max-num-seqs 4`, 262144 context.

## Problem

The long-prefill chunk handed to the attention backend is

    chunk = min(max_num_scheduled_tokens, mamba_state_block_size)

where `mamba_state_block_size` is the Mamba group's `kv_cache_spec.block_size`,
i.e. the recurrent-state checkpoint grid. The 75T Q8000 dense prefill route only
dispatches for `8000 <= max_seqlen_q <= 8192`, so the grid -- not the token
budget -- decides whether the fast prefill path is reachable at all.

That grid was not a free parameter. `unify_kv_cache_spec_page_size`
(`vllm/v1/core/kv_cache_utils.py`) exists because the KVCacheManager can only
allocate one page size for every group, so it scales the `block_size` of every
group whose page is smaller than the largest page:

    ratio = max_page_size // layer_page_size
    new_block_size = layer_spec.block_size * ratio

The largest page belongs to the higher-precision draft cache (DFlash2 runs
fp16 KV against an fp8_e4m3 target), so the Mamba and full-attention groups were
silently doubled. With `--block-size 4096` the grid therefore reached 8192 and
75T fired; with `--block-size 2048` it fell to 4096 and 75T was unreachable.
The finer block -- which is worth 23-29% more KV pool -- could not be combined
with the fast prefill path.

Instrumented scheduler output before the change:

    BLOCKPROBE block=2048 attn_block=2048 mamba_block=2048 kalign=2048 R=1636 mode=align
    GRIDPROBE  grid=4096 groups=[MambaSpec 4096 x6, FullAttentionSpec 4096 x2, SlidingWindowSpec 2048]
    CHUNKPROBE new=8192 start=0 pend=98631 chunk=4096 grid=4096 budget=8192 block=2048

The scheduler asks for the full 8192; the Mamba align clamp cuts it to the grid.

## Change

Two coordinated edits, both inert unless `--mamba-block-size` is passed
explicitly, so existing deployments are unaffected.

1. `vllm/platforms/interface.py`, align branch: keep an explicit
   `--mamba-block-size` as the recurrent-state checkpoint grid instead of
   overwriting it with `CacheConfig.block_size`. It is asserted to be a multiple
   of `--block-size`; see "Rejected configuration" below for why that is a
   correctness requirement and not a style rule.

2. `vllm/v1/core/kv_cache_utils.py::unify_kv_cache_spec_page_size`: for a
   `MambaSpec` carrying an explicit grid, do not multiply `block_size` by the
   page-unification ratio. This is memory-neutral: the physical page is still
   padded to `max_page_size`, and align-mode Mamba memory is
   `page_size_bytes * (2 + num_speculative_blocks)`, which does not depend on
   `block_size`. `get_kv_cache_groups` now passes `vllm_config` through.

## Result

| `--block-size` | `--mamba-block-size` | grid | chunk | 75T | @100K prefill | KV pool |
|---|---|---|---|---|---|---|
| 4096 (previously shipped) | - | 8192 | 8192 | yes | 3426 tok/s | 818,142 |
| 2048 | - | 4096 | 4096 | no | 2954 tok/s | 1,058,133 |
| **2048** | **8192** | **8192** | **8192** | **yes** | **3336 tok/s** | **1,058,133** |

A unique-salt 100K-token prompt is used for every figure; an earlier comparison
that re-used prompts which were prefixes of one another read up to 2.9x fast and
is discarded. The 4096 arm measured 3295-3426 tok/s and 818,142-858,310 KV
tokens across runs, so the decoupled 2048 arm is equal on prefill (3336 falls
inside that band) and 23-29% larger on KV capacity.

Probes after the change:

    BLOCKPROBE block=2048 attn_block=2048 mamba_block=8192 kalign=2048 R=1636 mode=align
    GRIDPROBE  grid=8192 groups=[MambaSpec 8192 x6, FullAttentionSpec 4096 x2, SlidingWindowSpec 2048]
    CHUNKPROBE new=8192 start=0 pend=98628 chunk=8192 grid=8192 budget=8192 block=2048
    route: Q8000 core / Q8192 FP32 75T dispatch

Default-path regression, `--block-size 4096` with no `--mamba-block-size`
(unchanged by this work):

    BLOCKPROBE block=4096 attn_block=4096 mamba_block=4096 kalign=4096 R=1636 mode=align
    GRIDPROBE  grid=8192 groups=[MambaSpec 8192 x6, FullAttentionSpec 8192 x2, SlidingWindowSpec 4096]
    CHUNKPROBE new=8192 start=0 pend=98633 chunk=8192 grid=8192 budget=8192 block=4096
    GPU KV cache size: 858,310 tokens

## Rejected configuration: `--block-size 1648 --mamba-block-size 8192`

1648 is the smallest legal block (the Mamba page is 1636 attention-tokens' worth
of bytes, and the platform rounds that up to `16 * cdiv(1636, 16) = 1648`); it
measured the largest KV pool of all, 1,112,091 tokens, and the fastest prefill,
3551 tok/s. It also silently destroys prefix caching, because 1648 does not
divide 8192.

A cached prefix is restorable only when some length is simultaneously
block-aligned (so the KV blocks exist) and grid-aligned (so a recurrent-state
checkpoint exists). That requires `grid % block == 0`. Measured on the 1648 arm:
`vllm:prefix_cache_queries_total = 812,040` against
`vllm:prefix_cache_hits_total = 0.0`, and an identical repeated 232K-token
prompt still took 86.5 s, where a hit would have been about 1 s.

The in-tree example in `v1/core/sched/scheduler.py` obeys the same rule -- an
816-token recurrent-state block against 16-token hashes, 816 = 51 * 16.

## Prefix-cache reuse is quantised by the grid, not the block

Warm the cache with a salted base prompt, re-send it, then send base plus a short
tail, and read the `prefix_cache_hits_total` delta:

| prompt tokens | reuse on identical re-send | reuse on prefix + tail |
|---|---|---|
| 1,538 | 0 | 0 |
| 2,642 | 0 | 0 |
| 4,438 | 0 | 0 |
| 6,643 | 0 | 0 |
| 8,727 | 8,192 | 8,192 |
| 11,024 | 8,192 | 8,192 |

At grid 8192 the reusable prefix is quantised to 8192 even though the block is
2048, and a byte-identical prompt below 8192 tokens gets no reuse at all.
Because `chunk <= grid`, the reuse quantisation, the prefill chunk and the state
grid are one and the same knob. This is inherent to align-mode hybrid
scheduling, not an artefact of the coupling removed here, and it is why a
smaller block buys KV capacity only. Greedy outputs on a cache hit matched the
cold outputs in every case, so the decoupled grid does not corrupt results.

## Adopted default

`scripts/serve_qwen38_27b_nvfp4_v100.sh` pins
`--block-size 2048 --mamba-block-size 8192`. Re-derive if the prefill budget
changes: the grid wants to be a multiple of the block size that stays inside
[8000, 8192] so the 75T route keeps dispatching.
