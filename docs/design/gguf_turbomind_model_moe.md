# Canonical GGUF expert model integration

Prepare gate, up and down projections independently as TurboMind expert banks
before TP slicing. This permits mixed GGUF formats within an FFN and handles
Flash-Next Q2_0 K640 down weights as canonical group32 u2 before local K160
slicing. The model remains TP4; the expert method retains existing TP output
reduction and graph behavior. FP16 operands use FP32 accumulation.

The GGUF loader retains its prepared adapter and name map across weight
iteration and filtered rereads. Rebuilding the adapter loses the canonical
storage policy and needlessly expands Q2_0 to Q4_1. A full loader-entry
regression checks source type, mmap sharing and the retained policy.

Supported small batches reuse the existing fused expert route/unroute
operators. The capability report lists the routing policy and projection
operators; unsupported route geometry and mapped experts retain the existing
alignment and FP32 weighted reduction. Original-block joint gate/up and
calibrated down-vector model dispatch remain separate integration scopes.

## Validation

The ordinary package `1.5.2.dev749+g05b1e0ea1` passes four installed CPU checks
and four installed GPU checks on V100 with Torch 2.10 CUDA 12.8. Mixed
IQ3_XXS/IQ4_NL/Q2_0 FFNs cover M1/8/32/512, all four logical TP slices and
changed-input graph replay against official dequantization. Fresh import
confirms the shipped core without private overrides; all 213 dependencies
are compatible. This restored model boundary has no isolated speed claim.

Whole wheel SHA256:
`c3952f73c9df0f0a1d27e83d6436d2ea4389784f0b1dceaa9519edc35ce0eb54`.
Core SHA256:
`b82a3ef00ddf7abcd80284548474becb2c06cbaa15e3791881db30344db5e1df`.
