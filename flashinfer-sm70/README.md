# FlashInfer-SM70 Primitive Compatibility

This directory is an isolated Volta compatibility layer.  It does not enable
an attention backend or claim an attention performance improvement.

`include/flashinfer/attention/sm70/volta_mma.cuh` exposes native Volta WMMA
fragment types for A, QK-B, PV-B, and the FP32 accumulator.  Its load,
init/load-accumulator, MMA-update, and store helpers let a caller retain one
accumulator fragment in registers across a complete reduction.

The two update operations keep FlashInfer's `m16n16k16` logical layouts:

| Operation | Matrix layout | Function |
| --- | --- | --- |
| QK | Q row-major times K-transpose through a col-major B view | `mma_sync_m16n16k16_row_col_f16f16f32` |
| PV | P row-major times V row-major | `mma_sync_m16n16k16_row_row_f16f16f32` |

QK expects the physical K tile as row-major `[N, K]`; the row/col WMMA view
then computes `Q[M, K] * K[N, K]^T`.  PV expects V as row-major `[K, N]`.
The fragment overloads update `AccumulatorFragment` without touching C
memory.  The original pointer overloads remain available as functional
controls: `kInit` starts at zero and `kInplaceUpdate` loads C, but every call
also stores C, so those overloads must not be used in a multi-K hot loop.

The implementation uses `nvcuda::wmma` when compiled for `sm_70`, which emits
Volta `HMMA.884` rather than recreating an `m8n8k4` raw-HMMA decomposition.
It is intentionally a primitive probe only.

Run the harness from the repository root:

```bash
.venv/bin/python benchmarks/benchmark_sm70_flashinfer_volta_mma_micro.py
```

It requires an idle physical GPU 0 at exactly 1200 MHz before compilation and
execution.  The harness fixes code generation to `sm_70`, runs random and
alternating-sign inputs for both layouts at `M=N=16, K=256`, and performs 16
ordered K16 updates between one accumulator load and one final store.  It
compares every FP32 output word with an independent direct `nvcuda::wmma`
reference, requires zero PTXAS spills and zero SASS `LDL`/`STL`, checks for
256 emitted `HMMA.884` instructions per kernel, and reports paired timings.
Timing is observational only and is not an attention promotion or speedup
gate.
