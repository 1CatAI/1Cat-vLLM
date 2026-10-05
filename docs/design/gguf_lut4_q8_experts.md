# Canonical IQ4 expert gate/up with Q8_1 activations

Flash-Next's final expert layer combines IQ4_XS gate/up and IQ4_NL down.
The current fallback uses three canonical grouped GEMMs. The post-pinned-PLE
TP4/MTP4 trace measures approximately 173 microseconds for those three kernels.
The existing canonical integer dot reader already handles IQ4_NL's codebook
and FP16 group scales. IQ4_XS uses the same codebook after canonical scale
expansion, so a joint gate/up launch can reuse that reader.

The proposed route shares activation quantization, FP16 SiLU/multiply and
integer down/unroute with the existing small-M expert path. It adds no new
weight decoder and keeps the original canonical banks for unmeasured batches.
Admission will require the measured TP4 geometry, IQ4 codebook zero, group32
scales, FP16 activations and FP32 route probabilities. M=5 and M=20 will be
screened independently before default admission. No new environment switch is
introduced.

Operator qualification will compare with the canonical baseline on the real
final-layer banks, plus independent official dequantization and activation
oracles. Graph replay tests must change inputs and routes. The kernel uses
FP32 dot accumulation and the already qualified Q8_1 activation protocol.
No performance or model-quality result is claimed before these measurements.

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
second weight bank. Full model quality and latency remain pending.
