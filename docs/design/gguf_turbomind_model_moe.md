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
These are layer checks. Full Flash-Next logits, generation and throughput
are still pending.

## Full-checkpoint loading

The IQ3_XXS checkpoint loads with TP4 on V100 32 GB x4, CUDA 12.8,
Torch 2.10.0+cu128, FP16 activation/KV, FP32 SSM state, no MTP, eager
execution, max length 2048, batch budget 256 and four sequence slots.
Model weights occupy 17.74 GiB per rank. Loading takes 662–741 seconds;
engine profiling and warmup complete in another 69.61 seconds. Linear
projections select the canonical affine, LUT4 and lattice implementations.

The first generation harness stops before generation when serializing a
Transformers `BatchEncoding`. Its replacement explicitly requests token ID
lists and validates all four prompts outside GPU execution before rerunning.
No generation result is inferred from the successful initialization.

The combined wheel with packed PLE cascade admission has SHA256
`299b95c853ab7b2184a03406ec9d0d51fbaa6732e7debebdf32f0250f1315a86`.
Eight relevant Python modules match source, wheel and installed bytes; the
canonical core and corrected FA2 fingerprints remain unchanged. All 21
installed cascade configuration checks pass. The PLE CPU worker loads its
five checkpoint entries and verifies both materialized parameters.

The pinned llama.cpp CPU reference consumes identical prompt IDs and returns
`Paris`, `4`, `你好` and a coherent Chinese explanation. All four first-logit
vectors are finite. The explanation reaches the reference's 64-token limit,
so its greedy comparison is limited to that prefix. The installed TP4 run
completes all four requests with natural EOS. Paris and arithmetic match the
reference bodies. Translation correctly includes `你好` but differs from the
reference at its first token; the Chinese explanation is coherent and differs
at its second token. Token identity is diagnostic rather than a correctness
gate. Numerical comparisons and the broader quality suite remain pending.

## Numerical and performance measurement

Model Runner V2 supports raw log probabilities but rejects raw-logit mode.
Requesting raw logits currently falls back to V1, which is inappropriate for
the Qwen4Exp QSA/PLE state contract. The initial measurement attempt was
stopped before it produced performance data; its log is retained. Use the
V2-supported `raw_logprobs` mode and request the full vocabulary only for a
separate one-token request. Natural generation and timing requests do not
request log probabilities.

Compare the saved vectors to the same GGUF CPU reference after applying
log-softmax to reference logits. Report finite values, KL in both directions,
JS divergence, total variation, and centered-logit maximum absolute error,
RMSE, relative L2 and cosine. Centering removes the unobservable per-row
normalization constant; these are not absolute raw-logit measurements.
Do not require token-by-token identity. Operator reconstruction errors and
FP16 coefficient rounding remain separate from model distribution and quality.

The matched timing contract is TP4, FP16 activation/KV, FP32 SSM, no MTP,
graph execution, max length 33024, batch budget 8192, 16 sequence slots and
GPU memory fraction 0.85. Decode uses input/output 1024/128 at C1/C4/C8/C16;
prefill uses 8192/32768 tokens and one output token, with two measured repeats
after warmup. GGUF and native NVFP4 use identical frozen prompt IDs.
