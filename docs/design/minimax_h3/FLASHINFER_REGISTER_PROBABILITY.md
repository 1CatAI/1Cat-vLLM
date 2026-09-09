# FlashInfer SM70 probability fragments in registers

This change improves the explicitly selected FlashInfer backend. It does not
qualify an AUTO configuration or meet the campaign's >80 useful TFLOP/s/card
gate. The shared `sm70_attention.noncausal_attention` interface dispatches on
SM70 hardware, FP16 tensors and BSHD D128 layout, independently of model,
quantization and adapter identity.

## Arithmetic and execution

One warp owns 16 query rows and both logical 32-key halves of the existing
64-key tile. QK/PV accumulators, online maxima and denominators remain FP32.
Each original partial sum, XOR-2/XOR-8 reduction, left/right addition and
per-output MMA K order is retained. Probabilities cross the same FP16 rounding
boundary as before.

Six 32-bit lane exchanges convert the rounded probability pairs into Volta A
fragments. PV reuses each fragment across eight output fragments. Probabilities
no longer traverse shared memory, and the warp owns its softmax state without
cross-warp barriers. CTA barriers still protect K/V staging and reuse.

The CTA covers 192 queries with 384 threads. Cooperative prefetch rounds up
to three vectors per thread. Extra complete warps in the final prefetch group
skip rows outside the 64-key tile; the guard is warp-uniform before shuffles.
Valid-length padding, batches, heads and unaligned global-storage loads retain
their original handling. Explicit lane selects avoid addressable local arrays.

CUDA 12.8 emits 168 registers/thread, no stack or spills, and 84,992 dynamic
shared bytes. These resource counts describe the implementation; they are not
throughput measurements.

The FP16-accumulator layout shortcut in [FastAttention Appendix B](https://arxiv.org/html/2410.16663v1#A2)
is outside this campaign's precision contract. This implementation exchanges
already rounded probabilities while retaining FP32 accumulation.

## Development evidence

Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA toolkit 12.8.93,
V100 SXM2 32GB. Artifacts are under
`/data/minimax-h3/sm70-general-20260909/`.

- `attention-fi-register-probability-q192/probe.json`: 17 boundary lengths
  and the actual 34,551-token, 14-head H3 capture match the frozen FI binary
  bitwise. Sampled FP32-reference relative L2 is 0.000374 on the actual input.
- Seven alternating operator timings on the same GPU give medians
  183.124985 -> 160.701447 ms, 12.245% lower latency. Observed clocks remain
  1425-1432 MHz. This is an operator measurement only.
- Compute Sanitizer 12.8 memcheck, racecheck and synccheck each report zero
  errors on 12 boundary cases with two batches, three heads and storage offsets.
- The formatted native build passes 33 GPU numerical, unaligned-storage,
  independent-query, graph and non-H3 shared-interface checks. The affected
  CPU contracts and provenance/acceptance suite pass 46 checks, with seven
  GPU checks skipped in the explicitly device-masked CPU invocation.
- `fi-general-control-quality.json`: the current common prepared/residual
  path preserves the previously frozen FI final video/audio latents bitwise.
- `fi-register-denoise-summary.json`: one complete denoise warmup per
  implementation, followed by one unprofiled measurement per implementation
  in reverse order. Both use the same TP4 W8A16 LightX2V four-step v1.2 weights,
  five sigma points, captured conditioning/noise, column layout, exact residual
  sharding and zero persistent FP16 weight cache. All four final-latent pairs
  match bitwise on every rank.

| Complete-denoise control | Slowest-rank seconds | Useful TFLOP/s/card |
| --- | ---: | ---: |
| Previous FI kernel, common execution path | 67.303216 | 46.108983-46.109000 |
| Register-probability FI kernel, same path | 62.471266 | 49.675363-49.675382 |

The isolated kernel change reduces this complete denoise by 7.179%. Candidate
steps take about 15.59-15.64 seconds. The control excludes encoder, VAE and
packaging, and includes only one measurement per implementation. It is not
the required full-request warmup-plus-three acceptance. Existing FA query-128
remains faster in its separately recorded full-request measurements.

Full native media preservation, formal repeated request measurements,
independent official references and human review remain separate gates. The
campaign remains incomplete. No precision gate or sampler setting is relaxed.
