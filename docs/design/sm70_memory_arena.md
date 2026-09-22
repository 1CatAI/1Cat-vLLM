# SM70 graph scratch and DFlash peak memory

This change reuses paged decode scratch across row capacities on each CUDA
stream, reduces the native FP32-accumulated prefill score block from 16384 to
8192 tokens, and retains/concatenates DFlash auxiliary snapshots in the draft
projection dtype. The rollout gate allows at most 3% matched serving slowdown
and requires numerical and output-quality checks with CUDA Graph enabled.

## Scope and rollback

- `VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE=0` restores independent decode row
  buffers. The default is 1. Device, stream, heads, dimension, partition size,
  partial dtype, and rounded partition capacity remain separate; only row
  capacities share storage. Keeping partition capacities separate avoids
  multiplying the largest context by the largest batch seen on a stream.
  Captured generations remain alive after growth, including warmup allocations
  later referenced by a graph. Capture normally proceeds largest-first.
- `VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS=16384` restores the previous
  prefill score capacity; the new default is 8192. Set before worker startup.
  This preserves FP16 operands and FP32 accumulation but changes online-softmax
  merge order. Rebuild the source extension; no wheel or sidecar is required.
- `VLLM_DFLASH_COMPACT_AUX_HIDDEN=0` restores concatenate-then-cast. The default
  is 1. Auxiliary snapshots use the dtype already required by the loaded draft
  projection. The target hidden states, residuals, and their addition retain
  their existing precision. Only the retained copy is converted. Other draft implementations retain the
  existing tensor interface through a fallback.

No TP count, batch=1, or model quantization restriction is introduced. Sharing
scratch assumes serial execution on its owning CUDA stream, matching the
existing per-stream cache contract. Concurrent streams use separate storage.

## Evidence and status

Base: `4f8e5e674a1afa4be6e712fd735386a875385a5b` (`onecat/main`).
V100 SXM2 32GB, CUDA 12.8, Torch 2.10.0+cu128; source build, no preload.
Native manifests, allocation traces, and per-request artifacts are retained in
the task-local handoff.

A baseline TP2 allocation history reproduces the 2.109 GiB temporary peak:
800 MiB FP32 auxiliary concatenation, 400 MiB cast projection input, five
160 MiB auxiliary snapshots, and two 80 MiB output tensors. Direct concatenation
removes the 800 MiB intermediate without changing projection arithmetic.
Removing only the concatenation exposes another 2.064 GiB peak during the
target forward. Storing the five auxiliary copies in the already-required
FP16 projection dtype also removes 400 MiB of retained snapshots. This is an
allocation attribution, not yet a final serving-memory measurement.

The prior Graph inventory contains 2007.56 MiB of per-row decode buffers where
774 MiB accommodates the largest shape, suggesting 1233.56 MiB reclaimable per
rank. The score block reduction saves another 768 MiB per rank. Actual memory
and speed results will be added after matched serving validation; at fixed GUM,
freed non-KV memory can become additional KV capacity rather than physical free
memory.

13 decode arena tests pass on V100: FP16 and E4M3 KV, 6/12/24 heads, alternating
row shapes, separate streams, warmup-buffer capture, and subsequent growth.
CPU-only: 5 pass / 8 GPU skips. The combined initial GPU gate has 45 passes;
22 compact projection/snapshot tests pass, including compiled and Graph
execution, strided input, FP16/FP32 inputs, and unchanged target tensors.
15 relevant DFlash loading/contract tests and the context-pipeline GPU test
also pass. Q8192 at 256K is 1.92% slower with 8K blocks in the operator gate;
Q8000 is 2.22% slower. All 256K biased/periodic stress cases pass the FP32
oracle and exact replay. Combined serving gates remain pending; these results
do not establish a serving-speed claim.

PR 661 addresses capture-time allocation retention on growth; this change
primarily removes duplicate row-capacity allocations and additionally retains
warmup allocations used by capture. PR 660's prefill bridge growth policy is
outside this change. Neither overlapping PR is imported wholesale.
