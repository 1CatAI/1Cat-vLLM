# SM70 AWQ QPN single-token operator

## Kernel layer

`_C::awq_moe_qpn_m1_sm70_out` implements the native-group-32 Qwen3.8
TP4 routed-expert geometry. This layer registers an inference-only operator;
it does not select it in the model runtime or change any default route.

The quadpair-N Tensor Core dataflow is derived from the existing NVFP4 QPN
implementation and its retained `LICENSE.v100-skinny` notice. This is a
model-specific AWQ adaptation, not an enablement of a generic Skinny backend.

The two launches are:

1. W13: selected experts in original router order, CTA-local FP32 reduction,
   FP16 gate/up materialization, then FP16 SwiGLU intermediate output.
2. W2: FP32 dot products, per-route FP16 output materialization, then ordered
   FP32 router-weight accumulation into the FP16 output.

No selected-weight bank, input replication, checkpoint rewrite or persistent
weight copy is added. The operator consumes the existing prepared banks and
caller-owned intermediate/output buffers.

### Admission contract

| Argument | Shape | Type |
| --- | --- | --- |
| Input / output | `(1, 2560)` | FP16 |
| Intermediate | `(10, 160)` | FP16 |
| W13 prepared weight | `(512, 2560, 40)` | INT32 |
| W2 prepared weight | `(512, 160, 320)` | INT32 |
| W13 4-byte metadata | `(512, 80, 320)` | INT32 |
| W2 4-byte metadata | `(512, 5, 2560)` | INT32 |
| W13 3-byte metadata | `(512, 80, 320, 3)` | UINT8 |
| W2 3-byte metadata | `(512, 5, 2560, 3)` | UINT8 |
| Expert IDs / router weights | `(1, 10)` | INT32 / FP32 |

All arguments must be contiguous on the same SM70 CUDA device. Weight,
metadata and activation pointers require 16-byte alignment; IDs and router
weights require 4-byte alignment. Output/intermediate must not overlap each
other or any input. Negative or out-of-range expert IDs contribute zero;
duplicate valid expert IDs retain their separate router weights.

The 4-byte layout stores the existing FP16 scale and rounded FP16 bias.
The 3-byte layout stores a FP16 scale and UINT8 zero point, read scalarly in
this layer. Bias is reconstructed at the same FP16 boundary. Dequantization
retains `half_fma(q, scale, half(-zero * scale))`; replacing this with
`half((q - zero) * scale)` is not an equivalent rounding contract.

## Numerical boundary and tests

The CTA-local reduction changes FP32 summation order relative to the legacy
TurboMind split-K route. Bitwise equality to legacy AWQ is not promised, and
neither path is declared the mathematical reference merely because it existed
first. Full-model acceptance must examine fixed-prefix raw logits and paired
quality, separately from speed and free-running token-stream equality.

Run the portable prepared-layout test on a V100 native build:

```bash
.venv/bin/python -m pytest -q tests/kernels/test_sm70_awq_qpn_m1.py
```

It independently constructs prepared metadata/weight tiles for both layouts,
checks one-hot reads across K/group boundaries, an FP64 W2 dot reference with
explicit FP16 rounding allowance, changing CUDA Graph inputs, duplicate and
invalid expert IDs, aliased/misaligned arguments, and the registered fake op.
It requires SM70; a CPU skip is not a GPU test pass. Shape-specific kernel
tests do not by themselves establish full-model quality or throughput.
