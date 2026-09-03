# SM70 Qwen3.8 No-MTP Concurrency Optimization

## Scope

This work improves Qwen3.8-Flash-Next NVFP4 no-MTP decode scaling on TP4 V100.
It does not change MTP, sampling, KV-cache precision, or model weights.

Integration base: `onecat/main@45a58ab6749096248dc15b1263bdf5faf51f5c70`.

## Frozen baseline and targets

The existing single-request contract is 70 tokens/s. The measured aggregate
throughput from the same `max_num_seqs=16` engine is:

| Concurrency | Aggregate tokens/s | Efficiency vs. 70 tokens/s | Target |
| ---: | ---: | ---: | ---: |
| 4 | 163.727 | 58.5% | 238 tokens/s (85%) |
| 8 | 280.841 | 50.2% | 420 tokens/s (75%) |
| 16 | 441.229 | 39.4% | 728 tokens/s (65%) |

Raw baseline:
`.artifacts/qwen38_flash_next_no_mtp_concurrency/result.json`.

## Evidence-first sequence

1. Generalize the native NVFP4 QPN expert route from token counts 1 and 5 to
   exact no-MTP decode widths 4, 8, and 16. Compare the complete routed MoE
   operation with the existing grouped path, including numerical error.
2. Raise compact routed-slot coverage only if the 160-route C16 microbenchmark
   beats the dense 512-expert dispatch and passes the same numerical oracle.
3. Measure exact TP4 20/40/80 KiB collectives before adding push or fused-sum
   variants.
4. Screen small-M projection, HyperConnection, and GDN candidates only after
   the MoE and collective contributions are known.
5. Run one C1/C4/C8/C16 end-to-end acceptance test after admitted candidates
   have enough projected savings; do not repeatedly relaunch the full model.

## Acceptance gates

- A microbenchmark candidate must improve median CUDA Graph replay time at its
  production shape and must not regress any covered shape by more than 1%.
- Native NVFP4 weights and FP16 activations remain unchanged. Report bitwise
  equality, maximum absolute error, and relative L2 error against the current
  production route.
- Before default enablement, run long-output text health plus coding, tool-call,
  and structured-output quality checks. A speedup with a quality regression is
  rejected.
- Runtime selection must be capability- and shape-based. It must not bind the
  model to one concurrency, KV dtype, maximum sequence count, or scheduler
  setting.

