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
this is not an end-to-end result. All four ranks record 1290 MHz SM and
877 MHz memory clocks during the timed model requests below.

The previous qualified projection wheel measures complete rounds of
17.237 ms at 1K and 18.345 ms at 8K. The less-than-12 ms round objective is
still unmet. New model timings must include acceptance length and emitted
token latency instead of substituting a projection or boundary service sum.

## Normal-wheel model comparison

One source-complete wheel contains the projection planes and these boundary
changes. The two arms differ only in `pass_config.fuse_allreduce_rms`. Both
include the direct attention output and outer GDN boundary wiring. The
workload is TP4 on the full NVLink four-card host, FP16 KV, FP32 SSM,
FULL_AND_PIECEWISE graphs, max context 262144 and seven probabilistic draft
tokens. Sixteen matched prompts generate 600 tokens each with temperature
0.7, top-p 0.9, top-k 20 and seed 123. Exclude the first 20 rounds per prompt.
Timing requests ignore EOS for matched lengths; separate natural prompts
terminate normally. Every timed request has four-rank 1290/877 MHz samples.

| Input | Fusion off, ms/round | Fusion on, ms/round | On tokens/round | On ms/output token |
| --- | ---: | ---: | ---: | ---: |
| 1K | 17.090 | 16.623 | 3.015 | 5.539 |
| 8K | 18.190 | 17.720 | 3.001 | 5.935 |

The fused arm saves 0.467/0.469 ms per complete round. Compilation records
106 replacements per rank, compared with zero before the epsilon fix.
The remaining boundaries are not assumed fused: five expose an additional
collective consumer for draft auxiliary hidden states; other misses still
need graph-local inspection.

Compare only common autoregressive contexts: 126 logit rows remain and two
rows are discarded after input divergence. Mean KL is 6.393e-6, maximum KL
4.766e-5 and top-1 agreement 100%. Both natural prompts produce identical
answers and terminate normally; four concurrent requests produce reasonable
text. The C4 request is a health smoke, not a throughput comparison.

The source, wheel hash, numerical results, clock windows and emitted-token
statistics are retained in `data/gguf_collective_norm_20261007.json`.
The less-than-12-ms complete-round objective remains unmet.
