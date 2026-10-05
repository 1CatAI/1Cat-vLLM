# GGUF QPN projection batch measurements

The TP4 27B GGUF target now uses shared-activation QPN down projections and
joint qkvz/a/b launches, and twelve additional pure IQ3_XXS gate/up pairs.
The integrated normal package at main
`49ac1c58ffa8d91a3ef4b1ea068178362cf28b2f` is compared with the native a/b and
48-pair baseline at `5620850f8119a744d0c7d3d737c40f1255dde9a5`.
These are measured complete speculative rounds, including draft work,
communication, sampling and scheduling.

## Matched unprofiled comparison

Both runs use the same sixteen prompts: eight with exactly 1024 input tokens
and eight with exactly 8192. Prompt token IDs, runtime configuration and
sampling fields are checked for equality. Each prompt produces 600 speed-fixture
tokens; discard the first twenty output rounds, then average round latency and
emitted length equally across prompts. Pool interval time and emitted tokens
for output-token latency. Natural text checks run separately with EOS enabled.

Qwen3.8-27B GSQ-RCO IQ3_S target and Qwen3.8-27B DFlash2 Q8_0 draft run on
four NVLink-connected V100-SXM2-32GB GPUs. CUDA 12.8, Torch 2.10.0+cu128,
Python 3.12.14, 300W power limits, 1290MHz SM and 877MHz memory clocks during
measured generation. TP4, maximum length 262144, batch token budget 1024,
four sequence slots, memory fraction 0.9, FP16 activation/KV and FP32 SSM
state are fixed. The TP4 probabilistic draft proposes seven tokens.
FLASH_ATTN_V100, async scheduling and CUDA graphs remain enabled; prefix caching
is disabled. Sampling is temperature 0.7, top-p 0.9, top-k 20, seed 123,
with thinking disabled. Reduced-precision matmul accumulation is disabled.
The complete `dev53+g49ac1c58ff` wheel runs without private native libraries
or source overlays.

| Input | Baseline round ms | QPN batch round ms | Saving ms | Emitted tokens/round | ms/output token | Output tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 22.219657 | 20.639314 | 1.580343 | 2.914170 | 7.100283 | 140.839 |
| 8192 | 23.316529 | 21.734425 | 1.582104 | 2.993911 | 7.299748 | 136.991 |

Baseline emitted lengths are 2.919855 and 2.939016. Round speedups are
1.0766x and 1.0728x. At 1K the emitted length is almost unchanged;
the 8K output-token improvement also includes a slightly longer emitted batch.
Different FP32 partition orders can change sampled histories; no bitwise
model-output equivalence is claimed. Unqualified M and source/shape combinations
retain canonical execution, as covered by the operator regression tests.

TTFT is 345.72ms and 2692.71ms, compared with 344.45ms and 2687.07ms.
These include prefill and scheduling. This comparison does not separately
time pure prefill or change the 1024-token prefill budget.

All four startup reports agree: eight pure IQ3_S pairs, fifty-two mixed-path
pairs including twelve pure IQ3_XXS pairs, thirty-seven down projections and
forty-eight joint qkvz/a/b projections. Four pure IQ4_XS pairs remain canonical.

C4 produces four reasonable nonempty 96-token outputs in 7.399s, compared with
7.466s in the baseline. Both are execution checks including first-use setup;
this does not establish warm C4 throughput. The arithmetic health prompt returns
`391`, and the English prompt explains unit tests in one sentence. Both end
naturally. The retained [aggregate record](data/gguf_qpn_projection_batch_20261006.json)
contains workload fields, cohort statistics, route counts and quality results.

One failed startup attempt stopped at an obsolete 40-layer mixed-pair assertion
before any timed request. The corrected fixture requires the actual 52 layers
and verifies all four ranks' down/qkvz counts. Its retry uses the same installed
wheel, sampling and prompt token IDs. Startup and graph-capture time are excluded
from the steady measurements.

## Graph-linked decomposition

A single post-batch capture follows the same historical trace fixture:
1024 input tokens, 64 output tokens, maximum length 32768 and one sequence.
The trace fixture uses top-p 0.95; the sixteen-prompt unprofiled comparison
uses top-p 0.9. These contracts remain separate. Graph-linked results and
projection/tail/gap attribution will be recorded after that capture completes.

## Remaining shared integration boundaries

The target-owned output head is called once by target rejection and once by
draft proposal. The inputs are distinct and data dependent; the earlier trace
does not establish an exact way to merge those GEMMs. Each reads 198656000
canonical bytes per rank. Head arithmetic and sampling stay unchanged in this
projection batch.

The draft attention contract is FP16 KV, D128, eight query heads, two KV heads,
832-token pages and non-causal 2048-token windows. The current FP16 grouped
verifier is qualified for D256, six/one heads and full causal context.
Enabling its model guard alone cannot serve the draft. A window-aware D128
implementation or split-KV path needs separate operator/context checks.

The shared TP4 push all-reduce/Gemma-RMS compiler pattern is available, but
the model's direct-attention-output guard currently requires
`quantization == "compressed-tensors"`. The GDN outer-all-reduce switch
depends on that guard, leaving the GGUF collective inside the whole-layer
operator. The compiler consequently cannot see it adjacent to the following
norm. A shared implementation that admits operand capabilities can expose
this boundary without adding a separate GGUF collective or norm kernel.
The guard finding is a source inspection; no communication speedup is claimed.
