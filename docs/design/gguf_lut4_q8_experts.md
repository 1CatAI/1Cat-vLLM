# Canonical IQ4 expert gate/up with Q8_1 activations

Flash-Next's final expert layer combines IQ4_XS gate/up and IQ4_NL down.
The current fallback uses three canonical grouped GEMMs. The post-pinned-PLE
TP4/MTP4 trace measures approximately 173 microseconds for those three kernels.
The existing canonical integer dot reader already handles IQ4_NL's codebook
and FP16 group scales. IQ4_XS uses the same codebook after canonical scale
expansion, so a joint gate/up launch can reuse that reader.

The route shares activation quantization, FP16 SiLU/multiply and
integer down/unroute with the existing small-M expert path. It adds no new
weight decoder and keeps the original canonical banks for unmeasured batches.
Admission requires the measured TP4 geometry, IQ4 codebook zero, group32
scales, FP16 activations and FP32 route probabilities. M=5 and M=20 are
screened independently before default admission. No new environment switch is
introduced.

Operator qualification will compare with the canonical baseline on the real
final-layer banks, plus independent official dequantization and activation
oracles. Graph replay tests must change inputs and routes. The kernel uses
FP32 dot accumulation and the already qualified Q8_1 activation protocol.
Operator and model results are reported separately below.

## Operator results

A source-complete SM70 wheel passes 85 focused GPU checks, covering the new
IQ4 path, existing IQ3/IQ2 readers, integer down/unroute, and graph replays
with changed activations and routing. IQ4_NL canonical dequantization is exact;
IQ4_XS expanded scales match the official reader within relative error 0.001
on the seeded format fixtures.

The same-process cold ABBA uses the real final-layer TP4 rank-zero banks:
512 experts, local intermediate 160, hidden 2560, top-k 10. Routing and
activations are seeded synthetic inputs. Each timed replay evicts 32 MiB
before the complete expert chain. The candidate includes activation encoding,
joint gate/up with FP16 SiLU/multiply and routed Q8 encoding, and integer
down/unroute. The control includes production routing, three canonical GEMMs,
compiled activation glue, and unroute.

| Original M | Control, two arms (microseconds) | Candidate, two arms | Relative L2 | Unique-weight effective GB/s |
| --- | --- | --- | --- | --- |
| 5 | 229.38 / 223.23 | 70.66 / 70.66 | 0.01060 | 479.35 |
| 20 | 327.68 / 308.22 | 199.68 / 199.68 | 0.01101 | 564.23 |

Effective bandwidth divides unique canonical code/scale bytes by chain time;
it is not measured DRAM traffic. Operator savings are approximately 156 and
118 microseconds for this one layer, not an end-to-end model claim.

Dispatch uses the shared kernel capability framework and the existing opaque
actual-M boundary. Only M=5/20, FP16 activations, codebook zero/group32 IQ4
banks, IQ4_NL down, and the measured geometry are admitted. Other batches use
the canonical banks. The `sm70_gguf.lut4_expert_dp4a` field allows a matched
control without disabling existing IQ3/IQ2 routes. The candidate allocates no
second weight bank. Full model results follow below.

The [retained operator record](data/gguf_lut4_experts_20261006.json) contains
every timing arm, numerical screen, shape and native artifact hash.

## Whole-model qualification

A matched source-complete wheel compares only
`sm70_gguf.lut4_expert_dp4a`, with all existing IQ3/IQ2, coalesced dense, HC
and pinned PLE routes enabled in both arms. Flash-Next IQ3_S uses TP4 on four
V100-SXM2-32GB, FP16 MTP4, FP16 KV, FP32 recurrent state, FULL target graphs,
maximum length 9216, token budget 512, four sequence slots and 0.95 GPU memory
utilization. Torch is 2.10.0/CUDA 12.8. The measured source is `e03af7c791`;
subsequent integration adds the separately qualified 27B-only QKV path, which
does not admit Flash-Next's K2560 shape.

C1 uses I8192/O256 and trims the first/last eight eligible decode records.
Each unobserved cohort contains 35 intervals and emits 171 tokens; C1 output
tokens match exactly across the switch.

| Metric | Control | Candidate |
| --- | --- | --- |
| C1 mean, pooled unobserved cohorts (ms) | 18.9961 | 18.5337 |
| C1 median, before/after CPU observation (ms) | 18.6314 / 18.6341 | 18.4489 / 18.5409 |
| C1 emitted tokens per round | 4.8857 | 4.8857 |
| C4 mean round, I128/O600 per request (ms) | 44.7538 | 43.8531 |
| C4 median round (ms) | 43.9741 | 43.3398 |
| C4 emitted tokens per measured round | 9.4933 | 9.4350 |

The control's second C1 cohort includes a large outlier; it is retained.
The 0.4624 ms pooled-mean difference and 0.9007 ms C4 difference must not be
attributed entirely to this one layer. Median C1 gains are approximately
0.09–0.18 ms, consistent with the operator estimate. The 18.5 ms phase target
remains unmet by the candidate mean, by approximately 0.034 ms; the ultimate
15 ms target also remains open.

At 64 strictly matching saved prefixes, teacher-forcing mean KL is 0.0007155,
maximum KL 0.0085970 and top-1 agreement 63/64; all logits are finite. Eight
natural prompts with 600-token budgets give mean draft acceptance
45.5865% / 46.0660%, with paired bootstrap 95% change interval
[-1.3998, +2.7173] percentage points. Accepted length is 2.82346 / 2.84264
(change interval [-0.05599, +0.10869]). The short arithmetic and Chinese
explanation prompts stop naturally with identical tokens. Long natural
outputs differ and remain coherent under the approved Q8 activation contract.

Startup logs confirm the IQ4_XS/IQ4_NL integer route and routed-Q8 output
at M5 and M20. Natural generation also exercises canonical fallback at M25.

## Partition screening

The integrated source-complete wheel passes twelve changed-input IQ4 GPU
checks and 37 expert/QKV capability checks. Its normal extension has no
private RPATH or extra native dependency. The installed IQ4 operator and
new 27B QKV operator are both present.

Further real-bank cold ABBA screens eight and sixteen lanes per output row:

| Lanes | M5 chain (microseconds) | M20 chain (microseconds) | Relative L2, M5/M20 |
| --- | --- | --- | --- |
| 4, model-qualified | 70.66 | 199.68 | 0.01060 / 0.01101 |
| 8, operator screening | 66.56 | 185.34 | 0.01060 / 0.01101 |
| 16, operator screening | 73.73 | 205.82 | 0.01060 / 0.01101 |

Eight lanes save about four microseconds for the single M5 IQ4 layer; this
is insufficient to close the phase target alone. Sixteen lanes regress.
The default retains the four-lane schedule that has full model speed and
quality qualification. Eight lanes remain available for a subsequent matched
model batch; no gain from that unpromoted schedule is included above. The
[8-lane record](data/gguf_lut4_experts_lanes8_20261006.json) and
[16-lane record](data/gguf_lut4_experts_lanes16_20261006.json) retain all arms,
including changes in control timing between separate processes.
