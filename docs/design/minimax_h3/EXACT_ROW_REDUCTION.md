# Explicit SM70 local-row reduction

The shared `SM70ExactRowReductionPlan` interface is experimental and has no
automatic dispatch or H3 runtime selection yet. The ordinary residual path
continues to use FP32 all-reduce followed by a local-row slice. No configuration
has passed the campaign's >80 useful TFLOP/s/card and complete quality gates.

## Arithmetic and ownership

A conventional reduce-scatter changes FP32 addition order relative to the
existing all-reduce, and the earlier full H3 control failed numerical gates.
This interface instead classifies the native communicator's addition order
for the actual two-dimensional FP32 shape. Ten fixed finite probes distinguish
all 15 four-input binary addition trees. Unmatched or ambiguous elements reject
setup collectively. Calibration never reads model weights or model activations.

The CUDA implementation copies each rank's partial input into owned IPC
storage, reads only its destination rows from peers and evaluates the calibrated
tree with rounded FP32 additions. System release/acquire flags establish input
visibility and completion. The grid has 80 blocks; the interface requires
SM70 devices with at least 80 SMs, four distinct peer-accessible devices on
one host, and consistent CUDA visibility. All row-parallel adapter contributions
must already be included in the input.

The caller supplies an explicit budget covering persistent IPC buffers,
local output, one-byte arithmetic codes and calibration scratch. GPU buffers
belong to the plan, not a global shape cache. Returned tensors alias the plan's
output; consume them before the next call. Call `close()` collectively before
tearing down the TP group. Rank-dependent setup errors are exchanged over the
CPU group, and peer handles close before owners free their allocations.

The plan rejects another stream/device, autograd inputs, incompatible layouts,
CUDA Graph execution and epoch exhaustion. It does not silently change a
backend or precision. Callers retain their ordinary collective when a plan
is unsuitable. Only the shared operator is provided in this change; a model
must explicitly own its lifecycle before integrating it into requests.

## Validation and measured limits

Environment: Torch 2.10.0+cu128, CUDA toolkit 12.8.93, NCCL 2.27.5, four leased
V100 SXM2 32GB cards. Evidence root:
`/data/minimax-h3/sm70-general-20260909/exact-peer-reduction/`.

- The first arithmetic classification covers every element of the real
  34560x5376 projection. Three independent wide-range inputs and signed-zero
  controls match the native all-reduce bitwise.
- The source implementation's TP4 control covers seven shapes from 4x3 through
  34560x5376, including non-H3 DiT widths, tails and non-aligned storage offsets.
  All four ranks preserve bits for ordinary, wide, subnormal and opposing
  infinity inputs: 112 numerical cases. Mismatched shapes, insufficient
  per-rank budgets, different streams and closed plans are rejected.
- The prototype's independent four-rank memcheck fixture reports zero errors
  on every rank. An earlier NCCL-bearing fixture reported only initialization
  `cudaFuncGetAttributes` probes for unsupported kernels; NCCL explicitly
  skips that return code in its [corresponding source](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/enqueue.cc#L37-L38).
  The isolated fixture uses Gloo for coordination and an independent FP32
  arithmetic reference; no CUDA API error suppression was applied.
- Source operator medians, including the full input copy and device barriers:
  native 14.561–14.641 ms, peer rows 10.939–10.991 ms. These seven alternating
  measurements are communication diagnostics only.
- A separate artifact override at source `3280edbfcc` completes a full denoise
  warmup per implementation, then one measurement each: 59.717282 seconds
  native versus 58.228244 seconds peer rows, a 2.49348% reduction. Candidate
  useful throughput is 53.295148–53.295167 TFLOP/s/card. Every final video/audio
  latent bit matches across all four passes, and the baseline also matches
  the previously frozen FA query-128 control. Each candidate pass uses 400
  peer reductions. This is not the full-request warmup-plus-three protocol.
- The prototype's full native media control also preserves both final latents,
  all 124 RGB frames and PCM bitwise. SSIM and RMS ratio are 1; spectral cosine
  exceeds 0.99999999999998. Its captured cold request takes 94.269191 seconds,
  including 61.688596 seconds denoise. This is native preservation, not an
  independent official-model or human audiovisual review.
- That full request peaks at 19,732,554,240 PyTorch-allocated bytes/card plus
  743,180,800 persistent raw IPC bytes/card. Their sum is 20,475,735,040 bytes;
  driver/library overhead is additional. Do not report only the PyTorch number.

The prototype full-model evidence precedes the packaged plan's dynamic
calibration and setup guards. It is not substituted for full-model validation
of this final interface. Native integration, finalized-interface media controls,
formal repeated requests, TP/shape breadth and official/human quality gates
remain incomplete. No AUTO promotion is made.

Reproduce the operator control with an owned native GPU lease and
`torchrun --standalone --nproc_per_node=4
benchmarks/kernels/benchmark_sm70_exact_row_reduce.py --output <new-directory>
--full-shape`. The optional `--extension` pins an already-built library; the
report records its SHA256 plus benchmark, CUDA and shared Python source hashes.
