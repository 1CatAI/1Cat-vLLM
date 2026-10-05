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

One completed post-batch capture follows the historical trace fixture:
1024 input tokens, 64 output tokens, maximum length 32768 and one sequence.
The trace fixture uses top-p 0.95; the sixteen-prompt unprofiled comparison
uses top-p 0.9. These contracts remain separate. A prior initialization
attempt ran out of disk before profiler capture; no GPU trace was collected
in that attempt. Twelve full graph-linked rounds per rank remain after
transition filtering. Rank0 contains 281 target matrix launches per round,
compared with 412 in the previous ledger.

| Rank | Full GPU round ms | Target graph ms | After target ms |
| --- | ---: | ---: | ---: |
| 0 | 22.285763 | 18.382603 | 3.903159 |
| 1 | 22.335891 | 18.426844 | 3.909048 |
| 2 | 22.318533 | 18.414099 | 3.904433 |
| 3 | 22.328288 | 18.428447 | 3.899841 |

Rank0 decreases from 24.093242 to 22.285763ms. The target graph decreases
from 20.189812 to 18.382603ms, while the tail remains 3.903159ms
(previously 3.903430ms). These are profiler diagnostics, separate from
20.639314/21.734425ms in the unprofiled sixteen-prompt comparison.

### Target projection service and counterpart operators

Kernel service is summed and divided by call count. Weight bytes describe
unique streams per rank, including the padded B/A tile. The capture inventory
omitted joint-projection ParameterList children; qkvz bytes are reconstructed
from its ordered admitted source tuple and the exact stream lengths validated
by the installed operator. Down uses captured original-record parameter sizes
or the captured canonical code/stat streams. No second capture is needed.
Effective GB/s excludes activation, workspace, codebook and repeated traffic;
it is not an NCU DRAM measurement.

The counterpart column is a separate same-machine, 1290/877MHz, real-weight
cold-L2 graph operator measurement. It is NVFP4 for FFN and channel FP8 for
GDN/attention in the NVFP4 checkpoint. Profiling and event-timing regimes
remain distinct; their differences are not complete-round speedups. All
counterpart arms and source formats are retained in the aggregate JSON.

| Role | Calls/round | Trace us/call | ms/round | Bytes/call/rank | Effective GB/s | Counterpart operator us |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qkvz+a_b | 48 | 43.635 | 2.094499 | 9973760 | 228.6 | 35.840–35.840 |
| gdn_out | 48 | 25.319 | 1.215324 | 4218880 | 166.6 | 18.432–18.432 |
| gate_up.native | 60 | 61.873 | 3.712394 | 18423467 | 297.8 | 50.176–53.248 |
| down | 64 | 36.767 | 2.353109 | 10564480 | 287.3 | 26.624–26.624 |
| attention.q | 15 | 37.660 | 0.564895 | 7929856 | 210.6 | 30.720–31.744 |
| attention.k+v | 6 | 29.087 | 0.174520 | 1556480 | 53.5 | 20.480–20.480 |
| attention.o | 16 | 23.050 | 0.368798 | 4177920 | 181.3 | 18.432–19.456 |
| attention.k | 9 | 33.605 | 0.302447 | 728178 | 21.7 | 20.480–20.480 |
| attention.v | 10 | 32.162 | 0.321617 | 720896 | 22.4 | 19.968–20.480 |
| attention.q+k | 1 | 38.103 | 0.038103 | 9584640 | 251.5 | 31.744–33.280 |
| gate_up.canonical | 4 | 56.543 | 0.226172 | 25067520 | 443.3 | 50.176–53.248 |

QKVZ and a/b now share 48 launches, so no arbitrary portion of one kernel
is assigned to a/b alone. The 60 original-record gate/up pairs comprise
eight IQ3_S, forty mixed and twelve pure IQ3_XXS layers. The four remaining
canonical pairs are pure IQ4_XS. The 64 down calls mix 37 QPN and 27 canonical
layers; their aggregate is not the latency of one quantization type.
Attention retains 41 physical q/k/v launches across sixteen layers.

### Target auxiliary service

| Category | Calls/round | us/call | ms/round |
| --- | ---: | ---: | ---: |
| communication_or_reduction | 130 | 11.811 | 1.535369 |
| rms_norm | 129 | 6.114 | 0.788659 |
| GDN_state_or_gating | 144 | 8.437 | 1.214925 |
| other_target | 128 | 3.834 | 0.490786 |
| layout_or_copy | 95 | 4.772 | 0.453330 |
| attention | 32 | 31.940 | 1.022087 |
| unfused_FFN_epilogue | 4 | 3.744 | 0.014976 |

Layout/copy calls decrease from 143 to 95 and unfused FFN epilogues from
16 to four. Communication and normalization remain at 130 and 129 calls.

### Work after the target graph

The tail still contains 210 kernels per round. The two head calls remain
separate: target rejection and draft proposal with distinct hidden inputs.

| Category | Calls/round | us/call | ms/round |
| --- | ---: | ---: | ---: |
| other_tail | 139 | 5.857 | 0.814165 |
| draft_GEMM | 23 | 43.337 | 0.996741 |
| communication_or_reduction | 18 | 12.998 | 0.233964 |
| target_head | 1 | 274.744 | 0.274744 |
| sampling_and_sorting | 23 | 7.396 | 0.170115 |
| draft_attention | 5 | 103.965 | 0.519827 |
| draft_shared_head | 1 | 276.640 | 0.276640 |

Each head reads 198656000 canonical bytes per rank, giving 723.1 and 718.1GB/s.
The graph does not establish an exact merge of these data-dependent calls.
Draft attention remains five paged calls with only eight CTAs each.

### Idle intervals and closure

| Rank0 interval | Mean |
| --- | ---: |
| Target kernel interval union | 16.875967 ms |
| Target gaps between kernels | 1.506636 ms |
| Tail kernel interval union | 3.285967 ms |
| Tail gaps | 0.617193 ms |
| Pure idle between graph boundaries | 355.650833 us |
| Draft graph end to next target envelope | 214.756500 us |
| Last tail kernel to next target | 11.095833 us |

The draft-end envelope includes intervening head and sampling work. Only
355.651us of graph-boundary idle remains after subtracting kernel interval
unions; the final kernel-to-target gap is 11.096us. Service totals overlap
and are not added to close the round; interval unions retain their residuals.

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
