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

The normal CUDA12.8 SM70 extension and complete wheel build pass. At
1290/877MHz,300W, actual TP4 M8/N4352/K5120,16MiB cold-L2 graph ABBA:
layer36 IQ3_S/Q4_K native68.608us versus canonical80.896us; layer47
Q4_K/IQ3_S native68.608–69.120us versus canonical81.920us. Original
payload bandwidth is319.9–322.2GB/s. Native relative L2 versus official
FP32 GEMM is0.000507–0.000531. Runtime M512/8/1/5/16/20/32/8 and
graph checks pass; non-M8 outputs stay bitwise canonical.

Admit only these two measured SM70 FP16 M8/N4352/K5120 gate/up
orientations. Two layers save an estimated25.088–25.600us per M8 round,
not an end-to-end measurement. Source-sized retention is44,216,320
bytes/rank. Coverage reaches40/64 gate/up layers, including32/40 mixed
pairs and85.328% of mixed-pair bytes.
