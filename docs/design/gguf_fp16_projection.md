# Floating shards inside mixed GGUF projections

Qwen3.5 GGUF projections can contain quantized QKV/Z and floating B/A
shards in one merged linear layer. Restoring a pure floating linear layer
does not cover that mixed case. Its floating shards otherwise reach a
generic matrix multiplication independently of their small output width.

`GGUFPreparedProjection` now declares a floating-shard capability and calls
`vllm::prepared_gguf_fp16_projection` when its static descriptor is admitted.
The operator chooses the M8 candidate using actual runtime rows,
inside the opaque operation. A prefill-first graph cannot freeze that choice.

## Operator and numerical contract

The reused implementation is
`Sm70Fp16GemvSiluKernel.apply_out` in
`vllm/model_executor/kernels/linear/fp16_gemv_silu.py`.
It invokes the existing Triton `_fp16_gemv_silu_ranges_kernel`; there is no
new CUDA kernel, private extension or weight layout.

| Condition | Candidate capability |
| --- | --- |
| Source descriptor | F16 (1) or BF16 (30) |
| Stored operand | Already converted, contiguous FP16 matrix |
| Shape | N12/K5120 or N24/K5120 |
| Activations | FP16, matching input width/device |
| Hardware | SM70 |
| Runtime rows | Exactly M8, including flattened leading dimensions |
| Policy | Existing GGUF kernel policy enabled by default |
| Registration | Normal `vllm::prepared_gguf_fp16_projection` wrapper present |

Every rejected descriptor records a reason in the projection admission
report. The wrapper also checks the underlying row-kernel capability before
launching. Unsupported inputs and every other M retain matrix multiplication.
Rejected static descriptors retain the established GGUF fallback dispatch.

The row kernel receives `activated_columns=0` and divisor 1. It performs
FP32 products, accumulation and reduction, then rounds the projection result
to FP16. Its prefix SiLU is inactive. The existing loaded FP16 weights remain
unchanged, including the loader's BF16-to-FP16 overflow checks. This does not
combine scales, quantize activations or route QKV/Z through a new operator.

## Normal package dependencies

The row implementation is Python/Triton source shipped by normal vLLM
package discovery. Torch and the declared Triton/CUDA toolchain provide its
JIT compiler and driver libraries. Importing the normal GGUF preparation
module imports the wrapper module, which registers the operation and fake
implementation through `direct_register_custom_op`. No separate native
library registration, CMake target or external DSO is required.

The installed normal `1cat-vllm` 1.5.2.dev13 package RECORD includes the
existing `fp16_gemv_silu.py`. A candidate wheel must additionally include
`gguf_fp16_projection.py` and the updated GGUF preparation module. No installed
package was changed while preparing this implementation.

## Validation boundary

With Torch 2.10.0+cu128 and Triton 3.6.0, 21 CPU checks pass. They cover static
admission/rejection, disabled policy, actual-M selection after prefill,
fallback delegation, unchanged weight storage and a single opaque operation
in a dynamic full graph. CPU dispatch substitutes the established GGUF CUDA
fallback only in the delegation test; it does not establish GPU arithmetic.

The unchanged row implementation also compiles offline for SM70, N12 and
N24, K5120, block K256 and four warps. Both require 16 bytes of shared memory;
compilation did not initialize or launch a GPU kernel. This is a compiler
check, not a numerical or performance result.

The candidate has not passed GPU numerical, CUDA Graph replay or same-session
timing gates. It must pass those gates on real B/A shards and the normal
packaged runtime before integration. No speed improvement is claimed here.
