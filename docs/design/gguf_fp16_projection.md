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

## Real B/A operator verification

The standalone operator was tested on `blk.6.ssm_beta.weight` and
`blk.6.ssm_alpha.weight` from Qwen3.8-27B GSQ-RCO IQ3_S. The normal adapter
restores head order and converts the original BF16 matrices to FP16 before
taking TP4 rank-zero N12/K5120 shards. Their operands match official GGUF
Float dequantization followed by Half conversion bit for bit. Conversion's
maximum absolute error is 2.9802322387695312e-08; all values remain finite.

The installed normal wheel contains the wrapper, existing Triton row source
and declared dev14 native libraries. No source overlay or private DSO was
used. The updated native libraries include the CPU PLE registration. Extra
signed-pair native APIs are present but are not used by this floating operator.
The source tree does not include the signed-pair model dispatch changes.

Workload: local Tesla V100-SXM2-32GB, SM70, clocks 1290/877 MHz, Torch
2.10.0+cu128, Triton 3.6.0, FP16 activations and operands, FP32 accumulation.
This machine is distinct from the primary benchmark host. The test uses
real weights, synthetic seeded activations and standalone CUDA Graphs.

| Projection | Original mm, A arms (µs) | Row GEMV, B arms (µs) | Source bytes | Source-byte bandwidth (GB/s) |
| --- | --- | --- | --- | --- |
| b | 149.056 / 149.040 | 7.296 / 7.296 | 122,880 | 16.84 |
| a | 152.336 / 152.256 | 7.296 / 7.312 | 122,880 | 16.81–16.84 |

Each ABBA arm contains 84 timing samples, with a 16 MiB cache flush before
each call inside graph replay. Source-byte bandwidth is source storage divided
by kernel time; it is not a measured DRAM traffic counter. Both M8 projections
match official Float FP32 dot products rounded to Half and original mm
outputs bit for bit.

The runtime sequence M512 → M8 → M1 → M16 → M32 → M8 retains a single opaque
operation in each compiled graph. PyTorch creates a second graph for its M1
specialization. Both M8 points launch the existing row kernel; other points
retain matrix multiplication. Every captured/replayed output is bitwise
identical to its uncaptured counterpart. The unchanged M512 fallback has
relative L2 error 2.28e-05 against the official Float reference.

On the separate model trace, 96 old B/A calls occupy 11.5252 ms. Replacing
them with these standalone local timings projects about 0.70 ms of B/A work,
or 10.82 ms saved per round. That is a projection across machines, not an
end-to-end measurement. Primary-host operator confirmation and model-level
route/quality validation remain pending; no model speedup is claimed.

Reproduce the operator test from a normally installed wheel, holding the
shared GPU reservation and individual GPU lock:

```bash
python benchmarks/kernels/benchmark_gguf_fp16_projection.py MODEL.gguf \
  --operator-sha SOURCE_SHA --native-base-sha NATIVE_SHA --output result.json
```
