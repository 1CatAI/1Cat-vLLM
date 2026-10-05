# Integrated GGUF output and attention projection batch

The shared-A small output route covers 42 GDN and 13 attention output layers.
The joint Q/K/V route covers all sixteen full-attention layers. Each shape is
admitted only after same-clock real-weight cold-graph ABBA wins against
canonical execution. Other M and unsupported shapes retain bitwise canonical
fallback. FP32 scale products and accumulation are preserved.

## Matched model-level result

Real Qwen3.8-27B GSQ-RCO IQ3_S target with Qwen3.8-27B-DFlash2 Q8_0 draft,
TP4, four V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, Python 3.12.14,
300W, 1290MHz SM and 877MHz memory. All 472 clock samples over generation
show the same SM/memory clocks. FP16 KV, FP32 SSM, Flash-V100, CUDA graphs,
seven probabilistic draft tokens and draft TP4 remain fixed.

Use the same sixteen prompt token sequences and sampling: temperature 0.7,
top-p 0.9, top-k 20, seed 123, thinking disabled, 600 timing output tokens,
max length 262144, batched-token budget 1024 and max sequences four. Each
cohort has eight prompts. Omit the first twenty output rounds per prompt;
round and emitted-length means weight prompts equally, while output-token
latency pools the retained intervals and emitted tokens.

| Input | #984 round ms | Previous batch ms | Current round ms | Tokens/round | ms/output token | Previous-batch saving ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 22.220 | 20.639 | 19.512 | 2.903 | 6.735 | 1.127 |
| 8192 | 23.317 | 21.734 | 20.557 | 2.995 | 6.918 | 1.177 |

The output-token rates are 148.48 and 144.55 tokens/s. Mean TTFT is
345.02ms and 2692.89ms; prefill is reported separately from round latency.
All four TP ranks prepare 60 native gate/up pairs, 37 down, 48 joint QKVZ+B/A,
16 joint Q/K/V and 55 small output projections.

C4's four 96-token outputs are reasonable; its 7.703s cold smoke includes
first-use setup and has no matched warm-throughput baseline. Natural checks
stop normally: the arithmetic answer is `391`, and the English answer explains
what a unit test checks. Forced-length timing and natural-ending checks are
separate. This is one post-integration model run. Weight loading took about
366s before cache/graph preparation; startup time is excluded from decode.

## One post-batch trace

Use the established diagnostic contract: 1K input, 64 output, max length
32768, TP4, temperature 0.7/top-p 0.95/top-k 20/seed 123, first target GPU-node
boundaries. Thirteen complete rounds per rank are retained from one capture.
The profiling prompt and shorter output differ from the sixteen-prompt model
run; these GPU windows are diagnostic, not replacements for unprofiled TPOT.

| Rank | Full round ms | Target graph ms | Target union ms | Target gap ms | After target ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 21.043 | 17.129 | 15.766 | 1.362 | 3.915 |
| 1 | 21.091 | 17.175 | 15.721 | 1.454 | 3.915 |
| 2 | 21.083 | 17.165 | 15.696 | 1.469 | 3.918 |
| 3 | 21.102 | 17.185 | 15.727 | 1.458 | 3.917 |

Rank0 falls from 22.286 to 21.043ms per full
GPU round and from 18.383 to 17.129ms
inside the target graph. Target matrix calls fall from
281 to 256;
layout/copy calls fall from 95 to 37. The sixteen q/k/v concatenations and
forty-two GDN input head-layout copies are removed by the admitted routes.
The after-target envelope remains 3.915ms with
210 kernels per round. No draft-side speedup is claimed.

## Target projection ledger and checkpoint counterparts

Weight bytes are actual packed operand footprints from the loaded parameters,
including canonical stat views and padded floating B/A rows. They exclude
activation/workspace/codebook traffic and are not NCU DRAM measurements.
Effective GB/s is total weight bytes divided by summed kernel service time.
Mixed-format rows use per-call mean bytes and time. Joint q/k/v is one launch;
its duration is not divided among concurrently scheduled source tiles.

The NVFP4 checkpoint uses NVFP4 for FFN and channel FP8 for attention/GDN.
Corresponding kernels were measured on this machine at 1290/877MHz, M8 and
the same TP4 shapes. Keep checkpoint weight/format differences visible.

| Role | Calls/round | us/call | Bytes/call/rank | Effective GB/s | Checkpoint counterpart us |
| --- | ---: | ---: | ---: | ---: | --- |
| qkvz+a_b | 48 | 43.861 | 9973760 | 227.4 | 35.840 joint |
| gdn_out | 48 | 23.333 | 3806720 | 163.1 | 18.432 channel FP8 |
| gate_up.native | 60 | 61.735 | 18423467 | 298.4 | 50.176–53.248 NVFP4 |
| down | 64 | 37.139 | 10564480 | 284.5 | 26.624 NVFP4 |
| attention.q+k+v | 16 | 40.094 | 8650240 | 215.7 | 31.744–34.816 channel FP8 |
| attention.o | 16 | 21.704 | 3744000 | 172.5 | 18.432–19.456 channel FP8 |
| gate_up.canonical | 4 | 61.671 | 25067520 | 406.5 | 50.176–53.248 NVFP4 |

There are zero standalone B/A projections in the target graph; their rows
are part of the 48 joint QKVZ+B/A calls. Down combines 37 QPN and 27 canonical
layers; GDN out combines 42 QPN and six canonical; attention out combines
13 QPN and three canonical. The four pure IQ4_XS pairs remain canonical.
The prepared canonical-stream candidates also lose and are documented in
[the screening decision](gguf_qpn_canonical_streams.md). The common capability
report now names the pure IQ4_XS pair fallback `measured_route_not_faster`,
while unmeasured shapes keep their separate rejection reason.

## Target auxiliaries, tail and gaps

| Target category | Calls/round | us/call | Service ms/round |
| --- | ---: | ---: | ---: |
| communication_or_reduction | 130 | 11.951 | 1.554 |
| rms_norm | 129 | 6.191 | 0.799 |
| GDN_state_or_gating | 144 | 8.455 | 1.217 |
| other_target | 128 | 3.886 | 0.497 |
| layout_or_copy | 37 | 3.480 | 0.129 |
| attention | 32 | 32.150 | 1.029 |
| unfused_FFN_epilogue | 4 | 3.694 | 0.015 |

| After-target category | Calls/round | us/call | Service ms/round |
| --- | ---: | ---: | ---: |
| other_tail | 139 | 5.770 | 0.802 |
| draft_GEMM | 23 | 43.314 | 0.996 |
| communication_or_reduction | 18 | 14.223 | 0.256 |
| target_head | 1 | 271.681 | 0.272 |
| sampling_and_sorting | 23 | 7.374 | 0.170 |
| draft_attention | 5 | 105.077 | 0.525 |
| draft_shared_head | 1 | 277.080 | 0.277 |

Rank0 target inter-kernel gaps total 1.362ms;
after-target gaps total 0.617ms.
Pure gaps between graph envelopes total 357.803us.
Draft-attention graph end to the next target start is
216.078us, including heads/sampling and
work outside graph envelopes; the last tail kernel to target gap is only
10.570us. These definitions overlap;
service sums, unions and gaps must not be added as independent wall times.

| Head | Calls/round | us/call | Bytes/call/rank | Effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| target_head | 1 | 271.681 | 198656000 | 731.2 |
| draft_shared_head | 1 | 277.080 | 198656000 | 717.0 |

The full-vocabulary channel-FP8 QPN counterpart from the NVFP4 checkpoint
uses M8/N62080/K5120, FP16 operands/output and FP32 accumulation. Two cold
L2 graph arms at 1290/877MHz both take 412.672us, reading 317973760B per
rank (770.5 effective GB/s). The reference norm is 0.000208 and one thousand
graph replays are bitwise stable. This compares the same-shape operator;
it does not establish the NVFP4 model's selected end-to-end head route or
cross-checkpoint numerical equivalence.

## Draft attention, heads and communication decisions

The five draft attention calls have D128, local Q/KV heads 8/2, FP16 KV,
page832 and noncausal window `(2047,2047)`. The existing FP32-probability/PV
grouped verifier requires D256, heads 6/1 and full causal attention. Its
shape and mask conditions reject this draft contract. Relaxing admission
alone would be incorrect. A separate D128 grouped or split-KV specialization
can preserve FP32 arithmetic and the per-query window edges; it should be
screened at 1K/8K and window boundaries before model admission. No draft
attention route is changed in this projection batch.

The two head calls are target verification logits and draft candidate logits,
once each per round. Each reads 198656000B per rank through the canonical
Q4_K path. They consume different hidden states and have different dependency
positions; no exact shared evaluation or vocabulary reduction is established.

The target still has 130 communication/reduction calls and 129 RMS calls.
The shared AR+Gemma-RMS compiler pass reports zero replacements. In
`Qwen3_5DecoderLayer`, direct attention output and the GDN outer all-reduce
are still guarded by `quantization == "compressed-tensors"`; GGUF's GDN
collective remains inside the full-layer boundary. The existing common
implementation must expose that collective for GGUF before its following
add/norm can fuse. Retain this as shared structural work rather than adding
a separate GGUF collective/norm implementation. No unmeasured AR/norm saving
is included in the model result.

The [retained record](data/gguf_qpn_output_batch_20261006.json) includes all
cohort statistics, prompt-token hashes, rank windows, service tables and
same-clock checkpoint counterpart arms. Installed `dev59+g6d11576ee9` native
hashes match the complete normal wheel, with 648 native/build source inputs
identical to the qualified projection binary. The later fallback-report
change leaves kernel admission and numerical execution unchanged.

The normal `dev60+g5cbe3aea29` package also passes installed-native auditing
for the fallback-report update. Eighty-six existing pair/input/output checks
pass, as does the added full-admission rejection check. Interval support alone
does not admit a capability with a rejection reason. No new GPU route or
precision change is introduced by that reporting update.
