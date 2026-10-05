# Restore GGUF GDN head order during canonical preparation

Head-tiled GGUF output projections currently reorder the activation on every
forward. Affine canonical preparation can instead permute integer codes and
all group coefficients into the model head order before TurboMind packing.
The head dimension must divide into complete canonical groups. Other formats,
unsupported layouts and mixed projections with an unrestorable shard retain
the original input transform. Admission reports restoration and fallback
reasons. This changes no decoder, activation dtype or accumulation precision.

The validation fixture covers head-group factors2 and3, M1/M5/M20/M512,
official Q6_K dequantization and changed-input CUDA graph replay. Flash-Next
uses factor3 and head dimension128; the TP4 output shape isN2560/K1536.
Ten GPU checks and15 dense-admission CPU checks pass.

A CUDA graph microbenchmark cycles six distinct real Q6_K output weights,
with alternating A/B order over eight epochs. The six packed banks exceed
V100 L2 capacity. Existing canonical coefficient rounding versus official
GGUF dequantization has maximum absolute error2.0981e-4 and maximum relative
L2 error1.9500e-4 across the six banks. Permuting codes and coefficients adds
no new weight rounding. The different column reduction order produces these
FP16 projection differences:

|M|Input permutation(us)|Restored weights(us)|Maximum output difference|Maximum relative L2|
|---|---:|---:|---:|---:|
|1|25.140|15.383|1.2207e-4|4.2332e-5|
|5|17.694|15.252|2.4414e-4|2.8794e-5|
|20|20.629|18.274|2.4414e-4|2.1349e-5|

M5 saves2.441us per projection, or about0.088ms for36 projections; this is
an operator-based estimate, not measured full-model latency. M20 saves about
0.085ms for36 projections. The expected graph-node reduction is one copy per
restored projection. Model integration and graph-entry checks are pending.

Reproduce with `benchmarks/benchmark_gguf_head_layout.py MODEL.gguf OUTPUT.json`
using the installed ordinary wheel, CUDA12.8/Torch2.10cu128, V100/SM70, FP16
activation/weights and FP32 accumulation. The stock gguf reader cannot open
this mixed file because Q2_0 has type42; use the project's compatibility reader.
