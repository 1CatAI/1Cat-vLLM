# BailingMoeV3 integration

`BailingMoeV3ForCausalLM` is registered as a hybrid MLA/Kimi Delta Attention
model with tensor- and pipeline-parallel weight loading. This port reuses the
existing vLLM linear, KDA and MoE implementations; it does not introduce a
new checkpoint-specific engine default.

For FP8 configurations with `store_dtype: "mxfp4"`, routed experts use the
existing SM70-aware MXFP4 factory. Unquantized exclusions still take priority.
Ling's `routed_experts_quant_method: "mxfp4"` is mapped to this same storage
contract. SM70 requires the existing TurboMind MXFP4 support; other devices
use the factory's existing backend selection. This does not relabel FP8
weights as MXFP4: the checkpoint must contain the declared packed format.

## Compatibility details

- The currently supported KDA projection form requires `no_kda_lora=True`.
- `kda_safe_gate` explicitly selects the bounded sigmoid gate; the false
  case retains the softplus gate. These are different model formulas.
- Q/K/V convolution caches share one tensor; recurrent state is a second,
  FP32 tensor. Model and layer shape/dtype/copy contracts agree for both SD
  and DS layouts and include speculative convolution history.
- Abbreviated block-FP8 exclusions use module-segment suffix matching.
  Other FP8 models retain exact matching unless they explicitly request
  suffix semantics. Fused projections still require consistent precision.
- MLA rotary width uses `qk_rope_head_dim`; the already-applied rotary width
  is not halved again by checkpoint `partial_rotary_factor`.

## Audit and validation

AI-assisted integration of PR #468 against main `378a93a94302`, including
the earlier #548/#549/#550 fixes. The audit repaired safe-gate forwarding,
two-versus-four cache metadata, suffix-exclusion API compatibility and fused
MLA projection mapping.

23 focused tests pass, including CPU TP/PP weight mapping, quantization
dispatch and exclusions, FP64 gate references, SD/DS recurrent-state updates,
unrelated-state isolation and changing-input CUDA Graph replay on an owned
V100-SXM2-32GB. Shared quantization regression selection: 54 passed.
Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8 runtime, synthetic
small tensors with FP16 projections and FP32 recurrent state.

No full-checkpoint load, full-model quality benchmark or end-to-end inference
suite was run for this integration. These tests establish local numerical
and integration contracts, not a model-throughput or benchmark-score claim.
