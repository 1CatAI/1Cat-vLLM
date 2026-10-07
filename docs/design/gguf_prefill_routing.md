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
`kernel_config.sm70_gguf.prefill_routing` enables the admitted prefill path
by default and permits matched comparisons with the original implementation.

GPU qualification and the 32K prefill comparison are pending. CPU fallback
and dynamic-shape compilation tests pass; these do not establish GPU speed
or numerical equivalence.
