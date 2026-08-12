# SM70 skinny NVFP4 dense backend

## Scope

The skinny backend is an opt-in small-batch overlay for dense W4A16 NVFP4
linear layers on exact SM70 GPUs. It is model-agnostic: selection depends on
the quantization layout and GEMM shape, not on a Qwen model class.

Enable it with:

```bash
VLLM_SM70_QUANT_BACKEND=skinny vllm serve ... --dtype half
```

`skinny` extends the existing unified SM70 backend selector. Other quantized
formats continue to use TurboMind. NVFP4 uses this runtime frontier:

| Condition | Route |
| --- | --- |
| FP16, `M <= 3`, `K % 128 == 0`, `N % 8 == 0` | skinny SIMT |
| FP16, `4 <= M <= 16`, QPN-eligible shape | skinny QPN |
| Larger M or unsupported shape | TurboMind |
| BF16 activation | explicit FP16 conversion, selected route, BF16 output |

The adapter runs one eager comparison against TurboMind for each unique
`(N, K)` shape at load time. A non-finite result, exception, or relative error
above `3e-2` disables the skinny routes for the worker; TurboMind remains
available because its packed state is prepared unconditionally.

## Checkpoint contract

The backend consumes the standard native NVFP4 W4A16 representation:

- packed E2M1 codes: `uint8 [N, K / 2]`;
- FP8-E4M3 scales with group size 16: `[N, K / 16]`;
- one multiplicative global weight scale.

Both compressed-tensors and ModelOpt NVFP4 linear adapters can use the backend.
W4A4 checkpoints use it as a weight-only W4A16 fallback on SM70; their
activation scales are not consumed on Volta. A mixed ModelOpt checkpoint may
still have a higher device-capability requirement because its non-NVFP4 layers
need their own SM70 conversion or fallback. Converting such a checkpoint to an
SM70-compatible compressed-tensors artifact remains a separate checkpoint-build
step.

## Memory and fallback

For dense models the load-time representation contains:

1. the checkpoint-native codes/scales for SIMT;
2. one same-size fragment-order QPN copy;
3. the existing TurboMind packed fallback.

The native tensors are aliased before TurboMind preparation rather than
cloned. QPN therefore adds one NVFP4-weight-sized copy, not two. This profile is
appropriate for dense 27B-class models under tensor parallelism, but it must
not be applied wholesale to a large MoE expert bank.

MoE support should reuse the same CUDA operators and prepack ABI behind a
separate expert adapter with a bounded or lazy QPN cache. That keeps this dense
backend reusable while avoiding an unbounded duplicate of every expert.

## Provenance

The CUDA implementation is the production SIMT and QPN subset of the
MIT-licensed `Leonccaa/v100-skinny` project at commit
`f8194f7c3c9269fa74ee70b5029d53c20098f4c8`. Its license and pinned-source note
live beside the source in `csrc/quantization/skinny_nvfp4/`.
