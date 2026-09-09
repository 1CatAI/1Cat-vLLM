# Qwen3.8 Flash-Next SM70 default-path audit

The historical approximately 98 tok/s single-request baseline was measured
with FP16 activations/KV, native FP32 SSM state, NVFP4 expert weights,
TP4/PP1, 262144 context capacity, an 8192-token prefill chunk and no MTP or
prefix cache. It used explicit optimization switches.

This task checks whether ordinary model/capacity arguments can select that
same route. Model-scoped defaults and a default-route benchmark are under
validation; this document does not yet claim a new speed result.

The initial audit found checkpoint-FP16 GEMV, fused GDN input and fused HC
disabled by default. This also prevents the dependent auto dual-compile and
hybrid PLE route. Most MoE, router, QSA and TP4 push optimizations are already
enabled. Shared-expert overlap and MoE add/reduce still require opt-in.
