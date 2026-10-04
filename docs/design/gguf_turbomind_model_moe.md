# Canonical GGUF experts in the model lifecycle

The operator layer already supports canonical affine, LUT4 and lattice expert
banks. Connect that lifecycle to GGUF expert loading: transcode before TP
slicing, prepare each expert, retain independent projection formats and build
strided weight/stat pointers. Q2_0 down projections with local K=160 retain
unsigned two-bit codes and avoid the temporary Q4_1 storage expansion.

Group routed tokens by expert using GPU operations with fixed output sizes;
compute grouped gate/up/down projections and restore token order before the
existing FP32 routing-weight reduction. Kernel choice belongs to the shared
capability declarations, including calibrated lattice vector bands. Keep the
existing MoE scheduler, TP reduction and CUDA graph behavior.

This scope depends on dense preparation (#869), the Qwen4Exp adapter (#876)
and packed PLE integration (#877). Correctness checks will compare canonical
weights with official GGUF dequantization, sum four TP partials, and compare
mixed-family FFN outputs in eager and changed-input graph replay. Real model
concurrency and prefill measurements follow operator/layer validation.

Integration base: `99bbedf135190d3ae94c7164bdbcb241b904222c`.

## Initial installed layer checks

The normal wheel passes the source/member/installed checks for all changed
Python modules; the canonical core and corrected FA2 fingerprints are unchanged.
All five focused cases pass on V100: Q2_0 source retention, mixed IQ3_XXS gate /
IQ4_NL up / Q2_0 down FFNs at M=1/8/32/512, four TP partitions and changed-input
CUDA graph replay. The down bank uses u2/group32 with local K=160. FP16 operands
and FP32 accumulation are preserved; references use official GGUF decoding.
These are layer checks. Full Flash-Next model loading, logits and throughput
are still pending.
