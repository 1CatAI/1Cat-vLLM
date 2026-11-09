# SM70 GDN projection copy fusion

This is the copy-only extraction from PR #504, audited against main
`8d9c3518992059105d89939e8a46d75184505d8e`. The remaining HC/QSA experiments
in that PR are not included. AI-assisted implementation and review.

## Runtime contract

The existing opaque FP16 GDN input operation computes two unchanged linear
projections for M > 1, then makes four contiguous outputs. On SM70, packed
FP16 tensors of widths 4096 and 24 now use one copy kernel for those outputs.
M <= 1, other devices, dtypes and layouts retain their original paths.
No model identity or maximum batch-size condition is added.

`VLLM_SM70_GDN_BATCH_SPLIT_COPY` defaults to 1; set it to 0 to roll back
only this copy optimization. It does not enable the separately opt-in M1
GEMV or fused-input arithmetic. Ordinary GDN slicing remains unchanged:
that path does not necessarily need all four copies.

## Focused validation

- V100-SXM2-32GB, single owned GPU; Python 3.12.13, Torch 2.10.0+cu128,
  CUDA 12.8 runtime, FP16 inputs, TP4-local layer-0 checkpoint weights.
- 61 tests passed: projection admission, CPU fallback, role-specific
  GEMV plans, existing HC kernels, and changing-input CUDA Graph replay.
- The benchmark exercises all 65,536 half payloads, including NaNs,
  infinities and signed zero. All four branches preserve the payload.
- Alternating paired CUDA Graph timing, five samples, sixteen rotating
  copies of the same actual checkpoint layer; activations are synthetic.

| M | Two GEMMs + old copies (us) | Two GEMMs + fused copy (us) |
| --- | --- | --- |
| 2 | 66.682 | 54.042 |
| 4 | 62.960 | 52.528 |
| 8 | 63.504 | 53.181 |
| 16 | 64.970 | 54.358 |
| 32 | 66.762 | 56.093 |
| 64 | 82.038 | 72.794 |

These are complete-projection operator measurements, not model throughput
or a new end-to-end quality claim. Arithmetic is unchanged. The dedicated
benchmark and tests retain reproduction steps; full end-to-end testing was
not run for this extraction.
