# Original GGUF expert prefill calibration

Original expert storage retains a small-batch MMVQ route and admits MMQ only
for measured projection layouts. The calibration key records compute
capability, quantization type, input/output dimensions, expert count and
top-k. It does not depend on a model name. Future layouts need operator
accuracy and timing checks before adding a range.

The 2026-10-23 screen used real GLM-5.3-Flash UD-IQ4_XS `blk.3` tensors:
288 experts, top-k 8, IQ3_S gate/up (K=4096, N=2048), and IQ4_XS down
(K=2048, N=4096). Samples covered 32, 64, 128, 256, 384 and 768 source
tokens, with uniform routes over all experts and skewed routes over 16.
The devices were V100 SXM2, V100 SXM3, RTX 3090 and RTX 4090. The SXM3
sample retained its deployment limit of 100 W and a 950 MHz clock ceiling.

| Compute capability | Source-token range admitted |
| --- | --- |
| SM70 | 384–768 |
| SM86 | 384–768 |
| SM89 | 256–768 |

Down projections receive eight routed rows per source token; their admission
bounds include that factor. Smaller batches, other types/shapes, unmeasured
hardware and batches above the calibrated range retain their existing route.
The installed operator must still support the source type. Existing TP4
calibrations are preserved.

The screen compared outputs against both the existing MMVQ path and sampled
FP32 projections from independently dequantized weights. Both paths were
finite, and all sampled relative L2 errors were below 0.02. This tolerance
includes their existing activation quantization. Gate and down were timed
separately; these operator results are not whole-model speedup claims.

CPU contracts live in `test_gguf_measured_prefill.py` and
`test_gguf_original_prefill.py`. End-to-end validation must also cover natural
completion, cold/hot cache reuse, MTP and concurrent requests, because changing
floating-point reductions can change token choices even at temperature zero.
