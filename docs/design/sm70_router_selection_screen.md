# SM70 router selection screen

The Flash-Next MTP4 target trace contains 48 router top-k calls per round,
with 7.322 microseconds mean instrumented service per call. This is profiler
service rather than complete-round wall time.

An FP32 partial-key selection candidate was prepared before checking the
captured operand signature. All eight retained router specializations load
FP16 logits; M5/M10 already use lossless half keys and top-16 selection.
The candidate therefore does not address the measured target hotspot and
was reverted before GPU testing. Runtime dispatch never enabled it. No
numerical result or speed improvement is attributed to this screen.

A subsequent launch-geometry screen should use the existing FP16 selector,
including tie, signed-zero and nonfinite behavior, and compare changed-input
graph replay before changing any default. FP32 projection accumulation and
normalization must remain unchanged.
