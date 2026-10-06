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

IQ3_XXS retains the original FP16 block coefficient. Multiplying it by 0.25
before packing introduces a second rounding for subnormal coefficients; the
reader applies that factor in FP32 together with the local scale and rounds
only the expanded group coefficient. Canonical restoration retains every
index and sign and reconstructs the same group coefficient.

Initial operator checks used real TP4 27B rank-0 shards on V100-SXM2-32GB
at 1290 MHz, CUDA 12.8 and Torch 2.10.0. With M8, graph replay and rotating
more than 48 MB of weight planes, down projections measured 20.3–22.3 µs
and GDN output projections 10.6–11.1 µs. Relative L2 error against official
GGUF dequantization was 3.4e-4–7.5e-4 across 24 role/type cases. These are
operator measurements, not model-level latency results.

All 16 ordered format combinations passed fused gate/up checks against the
official reference and 50 unchanged-input graph replays. Mixed IQ3_S/IQ3_XXS
GDN inputs with a/b passed 20 changed-input split-K graph checks; counters
returned to zero after every replay. IQ3_S, IQ3_XXS and compact IQ4_XS planes
restored their canonical packed codes and coefficient metadata bitwise.

The temporary original-record DMVQ reader is admitted only for TP4 IQ2_XS
and IQ2_S down matrices (N5120,K4352,M8,KW8,split1). Same-process cold-L2
ABBA measured 23.81 versus 32.32 µs for IQ2_XS and 25.12 versus 33.91 µs
for IQ2_S against canonical GEMM. Relative L2 errors were 2.9e-4.
Q2_K down (33.70 versus 28.68 µs) and IQ1_M gate (43.46 versus 32.72 µs)
retain their prior routes. IQ2_XXS gate measured 23.76 versus 31.40 µs as a
single matrix, but that result does not qualify an entire fused gate/up pair.

GDN lattice and LUT4 planes retain their original GGUF head order. M8 loads
activations in that order inside the shared-activation kernel; other M uses
the existing input transpose and canonical arithmetic. This avoids changing
prefill accumulation order while removing the M8 activation-copy launch.
Affine GDN projections keep the existing restored-weight layout.

`projection_plane_scope` selects `all`, `gated_pair`, or `iq3_xxs` for
same-wheel numerical comparisons. The latter selects whole projections that
contain IQ3_XXS, including their fused companion shards; it does not isolate
an individual reader inside a fused launch. Only `all` admits temporary IQ2
down readers. The default production scope is `all`.

IQ3 admission also checks the cancellation term used by the byte-to-half
conversion. Expanded scales above 63.96875 would overflow `1024 * scale`,
even if the actual weight remained finite; these layers retain canonical
storage. Replacement is atomic across a fused projection, so a rejected shard
never leaves a mixture of plane and canonical buffers for a legacy reader.
