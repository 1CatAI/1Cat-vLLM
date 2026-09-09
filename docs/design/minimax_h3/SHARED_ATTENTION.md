# Shared SM70 attention interface

`vllm.model_executor.layers.sm70_attention.noncausal_attention` exposes both
native SM70 dense attention implementations without importing the H3 model
package. H3's dense facade uses the same function. Historical extension ABI
names and H3 loader exports remain compatible with existing wheels; a default
call still invokes the original four-argument entrypoint.

```python
from vllm.model_executor.layers.sm70_attention import noncausal_attention

output = noncausal_attention(
    q, k, v, scale=128**-0.5, backend="FLASH_ATTN_V100", query_tile=128
)
```

Inputs are CUDA FP16 BSHD MHA tensors with head dimension 128; accumulation and
softmax remain FP32. No model name, quantization label or adapter identity is
required. FlashAttention supports different Q/K lengths, strided storage and
explicit query/key tiles. FlashInfer requires matching lengths and receives
contiguous inputs. Invalid scales, including values overflowing FP32, fail
before native loading. Unsupported dtype, hardware and shapes fail at the
CUDA entrypoint. There is no silent BF16/FP32 conversion or backend substitution.

Callers own masks, suffix padding and sparse geometry. The H3 facade still
slices valid tokens before dispatch and restores zero suffix padding; VSA
keeps its separate prefix, selected-block and learned-gate implementation.
AUTO qualification is a separate unfinished campaign requirement.

## Validation

Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8.93, V100 SXM2 32GB.
No CUDA source or binary changes accompany this interface extraction.

- `shared-attention-gpu.log`: 12 GPU checks pass. Both native backends and both
  FA query geometries preserve direct-entrypoint results bitwise for BSHD
  shapes `(1, 1537, 24, 128)` and `(2, 65, 8, 128)`, including strided inputs
  and a later increase in the online maximum. Sampled FP32 relative L2 is
  below 0.001. Additional checks cover unequal Q/K lengths, dtype rejection,
  H3 poisoned suffix padding and CUDA Graph replay with new input values.
- `shared-attention-cpu.log`: 18 scale, deployment, service and residency
  checks pass; nine GPU cases are deselected.

These tests exercise non-H3 DiT operator shapes, not a second complete model.
They establish interface preservation, not a new end-to-end speed result or
independent official quality acceptance. Raw logs and binary hashes are under
`/data/minimax-h3/sm70-general-20260909/`.
