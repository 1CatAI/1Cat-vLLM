# Canonical GGUF expert down dispatch

Connect the existing IQ4_NL/Q2_0 down-vector operator to canonical expert
banks. Central capabilities admit routed-row counts 10 and 50 at local
K160/N2560/E512 with FP16 activations. Other batches retain grouped GEMM;
M200 remains excluded because its prior operator screen regressed. An opaque
custom-op boundary selects using the actual routed rows during graph capture
and replay instead of freezing a prefill decision.

This changes model dispatch only. Canonical storage, the native decoder,
FP32 accumulation and TP reduction remain unchanged. The admission report
includes vector capabilities and fallback reasons.

The ordinary package `1.5.2.dev752+gb4a913b7a` passes twelve installed CPU
dispatch cases and ten V100 GPU cases with Torch 2.10 CUDA 12.8. Native vector
and GEMM fallback both compare against official dequantization with changed
inputs and stable graph replay. Mixed-format FFNs retain their four logical
TP slice coverage at M1/8/32/512. All 213 dependencies are compatible.

The earlier real-shape down-vector operator sweep projects 0.342 ms less
M5 service across the model's 39 IQ4_NL and nine Q2_0 layers. This restored
dispatch adds no new isolated speed measurement and does not turn that
projection into a model round gain.

Whole wheel SHA256:
`98b79bba892e7f2304386e0fc9eb757ad43ee4ebbc98b0b0f5fc6f53546ff494`.
Core SHA256:
`b82a3ef00ddf7abcd80284548474becb2c06cbaa15e3791881db30344db5e1df`.
