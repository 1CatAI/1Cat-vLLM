# SM70 graph scratch and DFlash peak memory

This change reuses paged decode scratch across row capacities on each CUDA
stream, reduces the native FP32-accumulated prefill score block from 16384 to
8192 tokens, and concatenates DFlash auxiliary states directly into the draft
projection dtype. The rollout gate allows at most 3% matched serving slowdown
and requires numerical and output-quality checks with CUDA Graph enabled.

## Scope and rollback

- `VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE=0` restores independent decode row
  buffers. The default is 1. Device, stream, heads, dimension, partition size,
  and partial dtype remain separate; only row capacities share storage.
  Captured generations remain alive after growth, including warmup allocations
  later referenced by a graph. Capture normally proceeds largest-first.
- `VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS=16384` restores the previous
  prefill score capacity; the new default is 8192. Set before worker startup.
  This preserves FP16 operands and FP32 accumulation but changes online-softmax
  merge order. Rebuild the source extension; no wheel or sidecar is required.
- `VLLM_DFLASH_COMPACT_AUX_HIDDEN=0` restores concatenate-then-cast. The default
  is 1. Auxiliary snapshots retain their original precision; conversion occurs
  at the same projection boundary. Other draft implementations retain the
  existing tensor interface through a fallback.

No TP count, batch=1, or model quantization restriction is introduced. Sharing
scratch assumes serial execution on its owning CUDA stream, matching the
existing per-stream cache contract. Concurrent streams use separate storage.

## Evidence and status

Base: `4f8e5e674a1afa4be6e712fd735386a875385a5b` (`onecat/main`).
V100 SXM2 32GB, CUDA 12.8, Torch 2.10.0+cu128; source build, no preload.
Task artifacts: `/data/minimax-h3/task-cache/memory-arena-20260922/artifacts`.

A baseline TP2 allocation history reproduces the 2.109 GiB temporary peak:
800 MiB FP32 auxiliary concatenation, 400 MiB cast projection input, five
160 MiB auxiliary snapshots, and two 80 MiB output tensors. Direct concatenation
removes the 800 MiB intermediate without changing projection arithmetic.
This is an allocation attribution, not yet a final serving-memory measurement.

The prior Graph inventory contains 2007.56 MiB of per-row decode buffers where
774 MiB accommodates the largest shape, suggesting 1233.56 MiB reclaimable per
rank. The score block reduction saves another 768 MiB per rank. Actual memory
and speed results will be added after matched serving validation; at fixed GUM,
freed non-KV memory can become additional KV capacity rather than physical free
memory.

13 decode arena tests pass on V100: FP16 and E4M3 KV, 6/12/24 heads, alternating
row shapes, separate streams, warmup-buffer capture, and subsequent growth.
CPU-only: 5 pass / 8 GPU skips. Native rebuild and combined end-to-end gates are
pending; these initial results do not establish a serving-speed claim.

PR 661 addresses capture-time allocation retention on growth; this change
primarily removes duplicate row-capacity allocations and additionally retains
warmup allocations used by capture. PR 660's prefill bridge growth policy is
outside this change. Neither overlapping PR is imported wholesale.
