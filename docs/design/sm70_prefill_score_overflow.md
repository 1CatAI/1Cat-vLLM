# SM70 compact score overflow recovery

Finite FP16 Q/K can produce nonfinite or saturated compact score storage.
Recover affected 64-query tiles from original Q/K using FP32 scores, retaining
centered/scaled V and restoring after FP32 normalization. Existing finite-score
rounding, sampled tail shifts, margin and cuBLAS mode remain unchanged.

Flags reset on each call and graph replay. Completed prefix workspace is reused,
without a second large score allocation or host readback. Admission is based on
actual score storage, not model names or deployment topology. The independent
finite-score precision changes remain under evaluation.

Regression tests cover prefix, final partial prefix, tail, 256K KV, sparse
exceptional query rows and changed-input graph replay against FP64 references.
