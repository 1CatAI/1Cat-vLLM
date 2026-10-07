# SM70 GGUF prefill routing

Large routed batches previously sorted 64-bit expert keys, sorted the route
permutation again to invert it, and materialized FP32 expert contributions
before reducing them. At a 16,384-token chunk and top-10 routing, those
contributions alone require 1.56 GiB for a hidden size of 2,560.

The prefill route sorts 32-bit expert keys stably, writes the inverse
permutation while gathering activations, and reuses the ordered FP32 weighted
reduction without creating the contribution matrix. The four independent
FP32 accumulators and their final reduction order match the existing Torch
oracle; multiplication and addition remain separate.

Admission requires SM70, contiguous FP16 activations, 512 experts, hidden
size 2,560, top-k at most 16, and 33 through 16,384 tokens. Small-batch routing
keeps its existing implementation. Other geometries retain the Torch path.
`kernel_config.sm70_gguf.prefill_routing` and `prefill_unroute` enable the
respective operators by default. Their independent controls permit routing
comparisons while retaining the memory reduction needed by large chunks.

Ten GPU comparisons covering 33 through 16,384 tokens and repeated expert
IDs match the Torch FP32 oracle exactly. The isolated 16K routing/reduction
graph takes about 3.6–3.9 ms, compared with 15.0 ms for the original pipeline;
peak allocated memory falls from 5.55 GiB to 1.72 GiB. These are operator
measurements, not model-level results.

The original model path exhausts memory during 16K warmup when restoring an
800 MiB expert-output matrix. The fused reduction removes that restoration
and its FP32 copies. Releasing the gathered activation matrix before the
down projection also allows its storage to be reused. The 32K model-level
comparison remains pending.
