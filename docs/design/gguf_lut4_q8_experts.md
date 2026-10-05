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
