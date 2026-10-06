# Shared-activation GGUF projection planes on SM70

The M8 projection route packs Q4_K, IQ4_XS, IQ3_S and IQ3_XXS into
coalesced planes consumed by a shared-activation Volta MMA kernel. It supports
ordinary projections, fused GDN input projections with floating a/b weights,
and gate/up projections with a SiLU-multiply epilogue.

Weights are reconstructed with FP16 group coefficients and accumulated in
FP32. Admission requires numerical checks, same-wheel model distribution
comparisons, and matched graph benchmarks. Other batch sizes retain the
existing projection route. Split-K workspaces and counters belong to each
layer and must remain stable across graph replay.

The implementation derives from the dmv11 projection and pack/pack3 layout.
Measured operator results, memory accounting and model-level results will be
recorded here before admission.
