# GGUF collective and norm boundaries on SM70

GGUF stores RMS epsilon as a float32 metadata value. For Qwen3.8-27B this
becomes the Python float `9.999999974752427e-7`; the SM70 TP4 fusion pattern
previously required the literal `1e-6`. The captured graph contains the
collective followed by the supported opaque norm boundary, but the scalar
constant mismatch prevents replacement. The same-machine GGUF model logs
report zero matches.

Register the push collective/norm pattern with `hf_text_config.rms_norm_eps`
and pass exactly that value to the replacement. Do not round metadata to a
different Python constant. Qwen3.5 GGUF also uses the existing direct
attention output and outer GDN collective boundary for the qualified dense
TP4 shape. The local projection remains FP16; the existing collective and
norm keep FP32 residual and reduction arithmetic. No new collective kernel,
precision mode, or environment variable is introduced.

## Validation

Ten CPU fake-tensor graph cases check matching HF and GGUF epsilon values,
residual retention, and rejection of a different epsilon. Six constructor
cases check the GGUF/ compressed-tensors boundary and the existing TP,
hardware, quantization and bias restrictions. Ruff and pre-commit pass.

A four-card V100-SXM2-32GB full NVLink microbenchmark uses CUDA 12.8 and
Torch 2.10.0+cu128. Sixteen distinct copies of an actual loaded rank-0 TP4 GDN
output projection precede the collective and norm. All ranks use this same
shard and generated activation/residual/norm weights, so this is a boundary
microbenchmark rather than a real four-shard model layer.

| Rank | Separate projection/AR/norm (us) | Fused boundary (us) |
| ---: | ---: | ---: |
| 0 | 21.710 | 19.134 |
| 1 | 21.727 | 19.118 |
| 2 | 21.739 | 19.163 |
| 3 | 21.765 | 19.186 |

The CUDA-event graph comparison interleaves separate/fused/fused/separate
three times. FP32 residual bits match on all ranks and normalized outputs
remain within one FP16 ULP of an FP64 reference. Multiplying the approximately
2.58 us boundary saving by 127 gives an estimated 0.33 ms per target round;
this is not an end-to-end result. Clock-window audit and normal-wheel model
admission remain pending.

The previous qualified projection wheel measures complete rounds of
17.237 ms at 1K and 18.345 ms at 8K. The less-than-12 ms round objective is
still unmet. New model timings must include acceptance length and emitted
token latency instead of substituting a projection or boundary service sum.
