# Original Q4_K/IQ3_S shared-activation pairs

Layers36/47 share TP4 N4352/K5120 and use both IQ3_S/Q4_K gate/up
orientations. Each original pair reads22,108,160 bytes/rank; together they
account for5.915% of mixed gate/up bytes.

Compose the existing Q4_K affine and signed-book IQ3_S readers in the same
shared-A M8 gated-pair kernel. Packet layouts, original scales, final FP16
operand formation, FP32 dot/reduction and fused gated epilogue are unchanged.
No second decoder or weight representation is added. Model dispatch stays
canonical pending actual-weight numerical and matched speed checks.

The Q4_K prototype benchmark also admits these two raw-operator orientations.
It checks official GGUF FP32 dequantization, runtime-M graph fallback and
cold-L2 graph ABBA with recorded clocks. Normal build and GPU numerical/speed
checks are pending; no end-to-end result is claimed.
