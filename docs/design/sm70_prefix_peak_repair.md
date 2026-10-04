# SM70 compact prefix peak repair

Prefix PV detects scores exceeding the sampled maximum and its safe exponent
range. Affected tiles rescan the complete maximum and recompute their partial
numerator and denominator before online merging. No host readback or graph
branch is needed; repair kernels exit immediately for unaffected tiles.

The existing score margin, V scaling, cuBLAS math mode, tail scan and FP16
score storage remain unchanged. Outlier flags reset for each prefix block and
graph replay. The independent FP32 score-storage recovery is tracked separately.

Regression tests cover dense and sparse unsampled unequal peaks at Q8000 and
Q8192, including changed-input CUDA Graph replay against FP64 references.
