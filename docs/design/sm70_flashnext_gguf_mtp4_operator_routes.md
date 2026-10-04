# Flash-Next GGUF MTP4 operator integration

The target verifier dominates the original Flash-Next IQ3_S speculative
round. Floating projection restoration, batched FP32 HC, packed vocabulary
projection and small-batch expert alignment are measured as separate
operators before complete-model integration. Operator service savings do not
predict emitted-token throughput when speculative acceptance changes.

## Workload

Flash-Next GSQ-RCO IQ3_S, FP16 MTP4, TP4 on four V100 SXM2 32 GB cards.
Driver 580.173.02, Torch 2.10 CUDA 12.8, Python 3.12.3. Activations and KV
are FP16; SSM state and kernel accumulation are FP32. The TP topology has
direct NVLink edges 0–1, 0–2, 1–3 and 2–3; diagonals use SYS.

The engine uses maximum length 8704, four sequence slots, batch budget 512,
memory utilization 0.9 and ring policy auto. Decode graphs are FULL; mixed
prefill/decode uses the resolved FULL_AND_PIECEWISE policy. C1 uses 8192
input tokens and 256 output tokens. C4 uses 128 input tokens and 1024 output
tokens per request. Timing cohorts use greedy sampling and ignore EOS for
fixed output length. Separate natural greedy prompts respect EOS. Each
candidate is measured three times; the original C1 has one repeat and the
original C4 has three. No profiler is attached to these measurements.

## Complete-round comparisons

The first candidate, source `39442cce7b`, combines floating restoration,
packed FP32 router, replicated batched FP32 HC and the merged GGUF runtime
dispatch correction. It retains dense FP16 vocabulary weights and legacy
expert alignment. The later candidate, source `5fb07b92b8`, additionally
extends GDN a/b row dispatch through M=32, prepares Q6_K packed vocabulary
projection and uses fused expert alignment/restoration. It retains the
existing expert GEMM implementations.

| Candidate | C1 round (ms) | C1 tokens/s | C1 full acceptance length | C4 round (ms) | C4 tokens/s | C4 full acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original ring auto | 41.499 | 75.212 | 3.036 | 65.624 | 262.269 | 3.830 |
| Floating/HC integration | 36.747 | 93.788 | 3.459 | 66.061 | 202.019 | 3.055 |
| Packed head and fused routing | 30.932 | 78.803 | 2.643 | 56.924 | 258.219 | 3.403 |

Round time is total measured engine time divided by steady intervals;
throughput is total emitted tokens divided by that same time. Full acceptance
length comes from the complete request's speculative counters and includes
the bonus token. These populations differ. The latest steady emitted tokens
per request and round are 2.4375 at C1 and 3.674745 at C4.

The latest C1 repeats each contain 80 steady intervals and 195 emitted tokens,
with 98 draft rounds, 392 drafted tokens and 161 accepted draft tokens.
C4 repeats each contain 196 steady intervals and 2881 emitted tokens across
four requests, with 1205 draft rounds, 4820 drafted tokens and 2896 accepted
draft tokens. Token IDs and speculative counters are identical within each
candidate workload across repeats.

The latest candidate reduces C1 round time by 10.567 ms, but reduced
acceptance limits the throughput improvement over the original to 4.8%.
C4 round time falls by 8.699 ms; throughput is still 1.5% below the original.
The first candidate's C4 regression is substantially reduced but the
nonregression check has not passed. This result does not isolate individual
operator effects or confirm a cause of the acceptance differences.

All four natural prompts have the same token IDs as the original ring-enabled
runtime and finish normally at EOS, with lengths 2, 2, 2 and 63 tokens.
The longer Chinese response gives a coherent explanation of Rayleigh
scattering. Fixed-length timing trajectories differ between candidates;
natural-output agreement must not be claimed for all ignored-EOS tokens.

## Route and artifact checks

The latest ordinary package is `1.5.2.dev538` from `5fb07b92b8`. Its wheel
SHA256 is `df4b0bc86258d594f8ef0806cff9722d59a8a653636929f083b8ca05fd075a07`.
The packaged core SHA256 is
`0a8f4f3a2cfd4153cbfff09ff009f0be1745f1fa9f8fc437250742894ca113e1`.
Native source matches the separately built batched HC artifact; the ordinary
wheel contains the Python route changes. Fresh-process provenance and
standard Torch/CUDA linkage were checked before launching the model.

Worker records admit `GGUFLMHeadMethod` for the Q6_K vocabulary shard
`[62080, 2560]`. Logs confirm FP32 replicated HC, a/b row GEMV at M=5/10/15/20,
router top-k and fused expert alignment at those batch sizes. The target
and MTP proposer share the packed vocabulary head. No new model trace has
been captured for these candidates.

## Follow-up

First check the fused output reduction against the original Torch FP32
operation on the admitted geometry. Its initial sequential sum differs from
Torch's four-accumulator ordering; a local operator comparison can remove
that discrepancy without another complete-model restart. This is an
implementation difference, not a demonstrated cause of acceptance loss.

The expert grouped-vector work reuses original-block decoders from #897.
The checkpoint's gate/up tensors include IQ3_XXS in 17 layers, IQ2_S in 20,
IQ3_S in 10 and IQ4_XS in one; filename alone cannot select a decoder.
Dense quantized projection tuning remains separate. Further model profiling
follows qualified floating-route integration.

## Joint experts and PLE integration

The next source-containing installed wheel is `1.5.2.dev560+g22e3ff90d`,
with the same Flash-Next IQ3_S, FP16 MTP4, TP4, KV/state dtypes, batch budget,
sampling and C1/C4 request lengths as the previous operator integration.
It adds joint original-block IQ2_S/IQ3_S gate/up, calibrated canonical
IQ4_NL/Q2_0 down vectors, exact Torch-order unroute reduction and batched
PLE n-gram IDs. Unmeasured points retain canonical dispatch; the slower
IQ2_S gate/up and down-vector M=20 points remain excluded.

| Candidate | C1 round ms | C1 tokens/s | C1 full acceptance length | C4 round ms | C4 tokens/s | C4 full acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous operator integration | 30.932 | 78.803 | 2.643 | 56.924 | 258.219 | 3.403 |
| Joint experts and PLE | 26.668 | 90.544 | 2.560 | 54.748 | 258.709 | 3.485 |

C1 saves 4.264 ms per round and improves throughput by 14.9% against the
previous integration; against the original ring-enabled model it saves
14.831 ms and improves throughput by 20.4%. The 26.668 ms round is below
the separately recorded 27.4 ms NVFP4 MTP4 reference, but that comparison
alone does not establish equivalent emitted-token throughput or quality.
C4 saves 2.176 ms against the previous integration while throughput changes
by +0.19%. C4 remains 1.36% below the original ring-enabled throughput; this
result does not establish a strict C4 non-regression gate. The 15–17 ms round
target is not reached.

Each of the three C1 repeats records 82 steady intervals and 198 emitted
tokens, or 2.414634 emitted tokens per interval. Full-request speculative
counters contain 100 drafts, 400 drafted tokens and 156 accepted tokens.
Each C4 repeat records 257 steady intervals and 3640 emitted tokens across
four requests, or 3.540856 per request and interval. Full-request counters
contain 1177 drafts, 4708 drafted tokens and 2925 accepted tokens. Token IDs
and speculative counters match across all three repeats in each cohort.
Acceptance counters cover the full request rather than only the steady
interval window and must not replace its emitted-token denominator.

Four natural greedy prompts finish normally: Paris, arithmetic, Chinese
translation and a 63-token explanation of Rayleigh scattering. All four
match original model token IDs. These are text-health checks, not a broad
quality-set result. Timing cohorts use fixed output lengths with EOS ignored;
natural prompts honor EOS.

Runtime logs confirm joint original-block type21/type22 at M=5, type21 at
M=20, both down formats at 50 routed rows, fused small expert alignment,
FP32 replicated HC, a/b row GEMV at M=5/10/15/20, router and QSA top-k,
and batched PLE n-gram IDs. The ordinary installed artifact passes 46 CPU
checks plus 42 GPU/mixed-bank cases, all 211 dependency checks and fresh
native import. No private DSO, preload or library override is needed.

Model storage is 24.18 GiB per rank. The profiler leaves 2.75 GiB for KV,
69,360 tokens; captured graphs add about 0.38 GiB. The additional original
expert banks are retained only for type21/type22; type18's measured M=5 gain
does not justify another 2.55 GiB per rank in this integration. There is no
new model trace in this measurement.

Whole wheel SHA256:
`50f9c856415b3ccbdf428d678bd7664c57ad523b1bc78112f2958aa8b100dc63`.
Loaded native `_C` SHA256:
`2259d5e8591b06c8fa58aa274cbaf3293f05e9c477fdd86b146ca10e9504589f`.
