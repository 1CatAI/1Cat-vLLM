# Flash-Next QSA and router execution structure on SM70

The historical target service totals are 1.63 ms for QSA/indexing and 1.10 ms
for routing. The corresponding 0.101 ms and 0.140 ms estimates account for
ideal operand traffic at 900 GB/s, not complete operator latency. They exclude
selection, synchronization, dependent reductions, and concurrent work. These
service totals come from a diagnostic trace and are not an additive partition
of the unprofiled 17.4018 ms/round baseline.

## Profile and structural candidates

The QSA total contains scoring (0.540 ms), sparse attention (0.638 ms), merge
(0.133 ms), pre-index work (0.105 ms), selection (0.180 ms), and expansion
(0.034 ms). The paged scorer launches a separate query axis: the observed
M5 grid is `(5, 39, 1)`. Five queries from the same request repeatedly load
the same compressed keys and each pads four indexer heads to an MMA tile.

The router total contains projection (0.714 ms) and selection (0.383 ms).
The projection overlaps the shared-expert branch. One representative layer
has a 13.568 us projection overlapping a 16.224 us shared gate/up, followed
by a 7.616 us selector. Isolating that projection gives about 7.2 us; moving
its overlap to another part of the MoE chain does not automatically save
the difference.

## Shared-key indexer

A single request's two to eight queries occupy the MMA output dimension
together. A CTA loads a stripe of 32 compressed keys; the four warps each
score eight keys against up to eight queries and four heads. This replaces
independent query grids without changing the stored keys, causal positions,
FP16 operands, FP32 scores, final selector, or sparse-attention computation.
The FP32 K-reduction order can differ slightly from the Triton reference.

The normal CMake extension is selected through `sm70_qsa_shared_key` and
reports both its runtime use and fallback reasons. Admission requires SM70,
FP16 Q and indexer cache, H4/D128, M2..8, and one request. Multiple requests,
M20, prefill, and other layouts keep the existing implementation. Dimension
strides are supported; aligned contiguous dimensions use vector loads.
The switch remains disabled until the full-model gates pass.

## Experiments

All entries below are isolated, same-process CUDA-graph ABBA measurements
on a V100-SXM2-32GB. They do not establish an end-to-end model speedup.
Router projections use 48 distinct weight tensors; QSA uses 12 distinct
layer caches with 2,496 compressed-key capacity, H4/D128, and M5.

| Candidate | Control ms | Candidate ms | Result |
| --- | ---: | ---: | --- |
| CUDA shared-key QSA scoring, 12 layers | 0.2210 | 0.1037 | Continue to model gates |
| Triton grouped-query scorer, first version | 0.215 | 2.222 | Reject: register spills |
| Triton grouped-query scorer, simplified | 0.216 | 1.172 | Reject: still slower |
| Router split-K producers and last-producer join, 48 layers | 0.3475 | 1.4590 | Reject: extra work and joins |
| Router scalar weight reuse across five queries, 48 layers | 0.3478 | 0.3937 | Reject: slower than MMA |
| Router repeated top-10 selection, 48 layers | 0.2031 | 0.2025 | Reject: no useful M5 gain |

The CUDA scorer's prototype maximum score difference was below 5.3e-6.
The split-K router was bit-exact. The scalar router had relative L2 error
1.94e-5 and maximum absolute FP16 output difference 0.001953125. Repeated
top-10 preserved IDs and differed by at most 1.2e-7 in normalized weights.
These numerical results alone do not admit a slower implementation.

Delaying shared-expert work until after router selection was also tested
using complete real-weight MoE chains, including both branches and their
join. Eight chained calls took 0.6337/0.6346 ms for IQ3_S,
0.6256/0.6307 ms for IQ3_XXS, and 0.5898/0.5914 ms for IQ2_S
(control/candidate). Outputs were identical. The later contention cancels
the router's isolated gain, so this scheduling change is not included.

## Validation and admission

The packaged scorer must pass causal masks, invalid pages, ties, sliced
dimensions, both position integer types, changing inputs during graph
replay, and the M20 fallback. The benchmark compares both scoring alone
and the score/selection/expansion chain. The final gate compares the same
wheel with only `sm70_qsa_shared_key` changed: C1/C4 ms per round, tokens
per round, target teacher-forcing agreement, natural outputs, and acceptance.
No model gain is claimed until that gate is recorded.

## External references

[DeepSeek TileKernels top-k](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/moe/topk_gate_kernel.py)
provides a useful stable-tie contract for small expert selection.
[FlashInfer selection](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/topk.cuh)
provides alternative radix selection and synchronization strategies. Their
algorithmic ideas require measurements at this workload's 512 experts and
five query rows; reducing comparison count alone did not improve M5 here.
